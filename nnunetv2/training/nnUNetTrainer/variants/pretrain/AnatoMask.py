from pprint import pformat
from typing import List
import sys
import torch
import torch.nn as nn
from timm.layers import trunc_normal_
import math
import encoder3D
from decoder3D import LightDecoder
import torch.nn.functional as F
import numpy as np

def monte_carlo(model, inp, T):
    """
    model: STUNET with dropout layer
    inp : input tensor of (B, 1, H, W, D)
    T : number of monte carlo samples
    Returns : List(len=T) of logits tensors, each of list(len=5) for multi-resolution output, each of shape (B, 1, H, W, D) 
    """
    model.train()
    logits_list =[]
    with torch.no_grad():
        for _ in range(T):
            logits = model(inp)
            logits_list.append(logits)
    return logits_list

def calculate_softmax(logits):
    """
    logits: Tensor of shape (B, C, H, W)
    B: Batch size, C: Number of classes, H: Height, W: Width
    Returns: Softmax probabilities (B, C, H, W)
    """
    return F.softmax(logits, dim=1)

# 2. 픽셀 단위 조건부 엔트로피 계산
def calculate_conditional_entropy(probabilities):
    """
    probabilities: Tensor of shape (B, C, H, W)
    Returns: Conditional entropy map of shape (B, H, W)
    """
    # Avoid log(0) by adding a small epsilon
    epsilon = 1e-8
    log_probs = torch.log(probabilities + epsilon)
    entropy = -torch.sum(probabilities * log_probs, dim=1)  # Sum over classes
    return entropy  # Shape: (B, H, W)

def calculate_epistemic_uncertainty(logits_list):
    '''
    logits_list : list of len T, each element least of len 5, 
                    each element tensor [b,1,h,w,d]
    output : [b,1,h,w,d]
    '''
    new_list = [i[0] for i in logits_list] # 해상도 제일 높은 seg output만 사용
    logits_stack = torch.stack(new_list, dim=0)  

    # Apply Sigmoid to convert logits to probabilities
    probs_stack = torch.sigmoid(logits_stack)  

    # Mean and variance of probabilities across T samples
    mean_probs = probs_stack.mean(dim=0)  
    var_probs = probs_stack.var(dim=0)  

    # Squeeze to remove the single channel dimension
    epistemic_uncertainty = var_probs.squeeze(1)  
    return epistemic_uncertainty

def calculate_aleatoric_uncertainty(logits_list):
    '''
    logits_list : list of len T, each element least of len 5, 
                    each element tensor [b,1,h,w,d]
    output : [b,1,h,w,d]
    '''
    new_list = [i[0] for i in logits_list] 
    logits_stack = torch.stack(new_list, dim=0) 
    # prob_samples = [calculate_softmax(logits) for logits in logits_stack] # 다중 class인 경우 softmax 사용
    prob_samples = torch.sigmoid(logits_stack) # binary class에서는 sigmoid 사용
    # entropy_samples = [calculate_conditional_entropy(probs) for probs in prob_samples] # 다중 class 인 경우 
    entropy_samples = - (prob_samples*torch.log(prob_samples+ 1e-8) +
                        (1-prob_samples)*torch.log(1-prob_samples+ 1e-8))
    aleatoric_uncertainty = entropy_samples.mean(dim=0)
    return aleatoric_uncertainty


class SparK(nn.Module):
    def __init__(
            self, sparse_encoder: encoder3D.SparseEncoder, dense_decoder: LightDecoder,
            mask_ratio=0.6, densify_norm='in', sbn=False, teacher_feature_dims=[512,256,128,64,32]
    ):
        super().__init__()
        # input size = 112
        input_size, downsample_ratio = sparse_encoder.input_size, sparse_encoder.downsample_ratio # (112,112,128)
        self.downsample_ratio = downsample_ratio # 16
        self.fmap_h, self.fmap_w, self.fmap_d = input_size[0] // downsample_ratio, input_size[1] //  downsample_ratio,  input_size[2] //  downsample_ratio # 7,7,8
        self.mask_ratio = mask_ratio # 0.6
        self.len_keep = round(self.fmap_h * self.fmap_w * self.fmap_d * (1 - mask_ratio))

        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder
        self.sbn = sbn
        self.hierarchy = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norm_str = densify_norm.lower()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()
        self.projection_head = self.build_projection_head(teacher_feature_dims)

        # build the `densify` layers
        e_widths, d_width = self.sparse_encoder.enc_feat_map_chs, self.dense_decoder.width
        e_widths: List[int]
        for i in range(
                self.hierarchy):  # from the smallest feat map to the largest; i=0: the last feat map; i=1: the second last feat map ...
            e_width = e_widths.pop()
            # create mask token
            p = nn.Parameter(torch.zeros(1, e_width, 1, 1, 1))
            trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)
            # create densify norm
            if self.densify_norm_str == 'bn':
                densify_norm = (encoder3D.SparseSyncBatchNorm3d if self.sbn else encoder3D.SparseBatchNorm3d)(e_width)
            elif self.densify_norm_str == 'ln':
                densify_norm = encoder3D.SparseConvNeXtLayerNorm(e_width, data_format='channels_first', sparse=True)
            elif self.densify_norm_str == 'gn':
                densify_norm = encoder3D.SparseGroupNorm(e_width, e_width, sparse=True)
            elif self.densify_norm_str == 'in':
                densify_norm = encoder3D.SparseInstanceNorm(e_width, sparse=True)
            else:
                densify_norm = nn.Identity()

            self.densify_norms.append(densify_norm)

            # create densify proj
            if i == 0 and e_width == d_width:
                densify_proj = nn.Identity()  # todo: NOTE THAT CONVNEXT-S WOULD USE THIS, because it has a width of 768 that equals to the decoder's width 768
                print(f'[SparK.__init__, densify {i + 1}/{self.hierarchy}]: use nn.Identity() as densify_proj')
            else:
                kernel_size = 1 if i <= 0 else 3
                densify_proj = nn.Conv3d(e_width, d_width, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                         bias=True)
                print(
                    f'[SparK.__init__, densify {i + 1}/{self.hierarchy}]: densify_proj(ksz={kernel_size}, #para={sum(x.numel() for x in densify_proj.parameters()) / 1e6:.2f}M)')

            self.densify_projs.append(densify_proj)
            # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
            d_width //= 2

        print(f'[SparK.__init__] dims of mask_tokens={tuple(p.numel() for p in self.mask_tokens)}')


    def mask(self, B: int, device, generator=None): # masking randomly
        h, w, d= self.fmap_h, self.fmap_w, self.fmap_d # shape of 7,7,8
        idx = torch.rand(B, h * w * d, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)  # (B, len_keep= HxWxDx0.4) # (4, 157)
        return torch.zeros(B, h * w * d,dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, h, w, d)


    def mask_uncertainty(self, B: int, device, uncertainty_map1:torch.Tensor, uncertainty_map2:torch.Tensor, mask_ratio): # masking based on uncertainty
        """
        Args:
            B: Batch size
            uncertainty_map: (B, 1, 112,112,128)

        Returns:
            mask: Boolean mask of shape (B, 1, fmap_h, fmap_w, fmap_d).
        """
        epistemic_feature_tensor = F.avg_pool3d(uncertainty_map1, kernel_size=(16,16,16), stride=(16,16,16))
        aleatoric_feature_tensor = F.avg_pool3d(uncertainty_map2, kernel_size=(16,16,16), stride=(16,16,16))

        b,c,h,w,d = aleatoric_feature_tensor.shape # out_tensor의 shape는 (4,1,7,7,8)로 예상

        ep_tensor_flat = epistemic_feature_tensor.view(B, -1)
        al_tensor_flat = aleatoric_feature_tensor.view(B, -1)

        topk = int(0.5*ep_tensor_flat.shape[1])
        _, ep_topk_idx = torch.topk(ep_tensor_flat, k=topk, dim=1, largest = True)
        _, al_topk_idx = torch.topk(al_tensor_flat, k=topk, dim=1, largest = True)
        union_idx = torch.unique(torch.cat([ep_topk_idx, al_topk_idx], dim=1), dim=1)
        union_idx[:int(mask_ratio*ep_tensor_flat.shape[1])]

        mask = torch.ones(b, h*w*d, dtype=torch.bool, device=device)
        mask.scatter_(dim=1, index=union_idx, src=torch.zeros_like(union_idx, dtype=mask.dtype))

        return mask.view(b,1,h,w,d)
    
    
    def mask_single_uncertainty(self, B: int, device, uncertainty_map:torch.Tensor, mask_ratio): # masking based on uncertainty
        """
        Args:
            B: Batch size
            uncertainty_map: (B, 1, 112,112,128)

        Returns:
            mask: Boolean mask of shape (B, 1, fmap_h, fmap_w, fmap_d).
        """
        uncertainty_feature_tensor = F.avg_pool3d(uncertainty_map, kernel_size=(16,16,16), stride=(16,16,16))

        b,c,h,w,d = uncertainty_feature_tensor.shape # out_tensor의 shape는 (4,1,7,7,8)로 예상

        tensor_flat = uncertainty_feature_tensor.view(B, -1)

        topk = int(0.6*tensor_flat.shape[1])
        _, topk_idx = torch.topk(tensor_flat, k=topk, dim=1, largest = True)
        topk_idx[:int(mask_ratio*tensor_flat.shape[1])]

        mask = torch.ones(b, h*w*d, dtype=torch.bool, device=device)
        mask.scatter_(dim=1, index=topk_idx, src=torch.zeros_like(topk_idx, dtype=mask.dtype))

        return mask.view(b,1,h,w,d)
    
    

    
    def mask_intensity(self, B:int, device, uncertainty_map:torch.Tensor, intensity=1, masking_ratio=0.6): # intensity masking based on uncertainty
        
        out_tensor = F.avg_pool3d(uncertainty_map, kernel_size=(16,16,16), stride=(16,16,16))
        b,c,h,w,d = out_tensor.shape # out_tensor의 shape는 (4,1,7,7,8)로 예상

        tensor_flat = out_tensor.view(B, -1)
        if intensity == 1:
            sorted_idx = tensor_flat.argsort(dim=1, descending=True)
            intensity_val = torch.linspace(0.0,1.0, steps = tensor_flat.shape[1], device=out_tensor.device)
            intensity_values = torch.zeros_like(tensor_flat)
            intensity_values.scatter_(dim=1, index=sorted_idx, src=intensity_val.unsqueeze(0).expand_as(sorted_idx))
            intensity_values
            return intensity_values.view(b,c,h,w,d)
        else:
            topk = int(masking_ratio*tensor_flat.shape[1])
            _, topk_idx = torch.topk(tensor_flat, k=topk, dim=1, largest = True)
            mask = torch.ones(b, h*w*d, dtype=torch.bool, device=device)
            mask.scatter_(dim=1, index=topk_idx, src=torch.zeros_like(topk_idx, dtype=mask.dtype))
            return mask.view(b,1,h,w,d)

    ##### generate epistemic and aleatoric uncertainty map ####
    # monte carlo 수행

    @torch.no_grad()
    def generate_mask(self, loss_pred, guide = True, epoch = 0, total_epoch = 200, generator=None, original_mask = None):
        h, w, d= self.fmap_h, self.fmap_w, self.fmap_d
        B, L = loss_pred.shape

        ids_shuffle_loss = torch.argsort(loss_pred, dim=1)  # (N, L)
        keep_ratio = 2/3
        ids_shuffle = torch.zeros_like(ids_shuffle_loss, device=loss_pred.device).int()
        len_loss = 0

        if guide:
            ### easy to hard
            keep_ratio = float((epoch + 1) / total_epoch) * 0.5
            ### hard-to-easy
            # keep_ratio = 0.5 - float(epoch / total_epoch) * 0.5

            ## top 0 -> 0.5
        if int((L - self.len_keep) * keep_ratio) <= 0:
            # random
            noise = torch.randn(B, L, device=loss_pred.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_shuffle2 = torch.argsort(noise, dim=1)
        else:
            for i in range(B):
                ## mask top `keep_ratio` loss and `1 - keep_ratio` random
                len_loss = int((L - self.len_keep) * keep_ratio)
                easy_len = int((L - self.len_keep)) - len_loss

                ids_shuffle[i, -len_loss:] = ids_shuffle_loss[i, -len_loss:]
                temp = torch.arange(L, device=loss_pred.device)
                deleted = np.delete(temp.cpu().numpy(), ids_shuffle[i, -len_loss:].cpu().numpy())
                np.random.shuffle(deleted)
                ids_shuffle[i, :(L - len_loss)] = torch.LongTensor(deleted).to(loss_pred.device)

                ids_shuffle2 = torch.zeros_like(ids_shuffle_loss, device=loss_pred.device).int()
                ids_shuffle2[i, -len_loss-easy_len:-len_loss] = ids_shuffle_loss[i, -len_loss-easy_len:-len_loss]
                temp = torch.arange(L, device=loss_pred.device)
                deleted = np.delete(temp.cpu().numpy(), ids_shuffle2[i, -len_loss-easy_len:-len_loss].cpu().numpy())
                np.random.shuffle(deleted)
                ids_shuffle2[i, :(L - easy_len)] = torch.LongTensor(deleted).to(loss_pred.device)

        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # generate mask: 1 is keep, 0 is remove
        mask = torch.zeros([B, L], device=loss_pred.device, dtype=torch.bool)  # Changed from ones to zeros
        mask[:, :self.len_keep] = 1  # Changed from 0 to 1
        # unshuffle to get final mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        ids_restore2 = torch.argsort(ids_shuffle2, dim=1)
        easy_mask = torch.zeros([B, L], device=loss_pred.device, dtype=torch.bool)  # Changed from ones to zeros
        easy_mask[:, :(self.len_keep + len_loss)] = 1  # Changed from 0 to 1
        # unshuffle to get final mask
        easy_mask = torch.gather(easy_mask, dim=1, index=ids_restore2)
        return mask.view(B, 1, h, w, d), easy_mask.view(B, 1, h, w, d)

    # 내가 작성한 함수
    def forward_encoder(self, inp_bchwd: torch.Tensor, active_b1ff=None):
        # step1. Mask
        encoder3D._cur_active = active_b1ff  # (B, 1, f, f, f)
        active_b1hwd = active_b1ff.repeat_interleave(self.downsample_ratio, 2).repeat_interleave(self.downsample_ratio,
                                                                                                 3).repeat_interleave(
            self.downsample_ratio, 4)  # (B, 1, H, W)

        
        masked_bchwd = inp_bchwd * active_b1hwd # 둘다 ([4,1,112,112,128])


        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        fea_bcffs: List[torch.Tensor] = self.sparse_encoder(masked_bchwd)
        fea_bcffs.reverse()  # after reversion: from the smallest feature map to the largest
        return fea_bcffs
    
    def build_projection_head(self, feature_shapes):
        projection_layers = nn.ModuleList()
        for c in feature_shapes:
            # in_channels = 2*c
            in_channels = c
            out_channels = c
            projection_layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=1))
        return projection_layers
    
    def forward_embd_loss(self, embd1, embd2, loss_ftn):
        loss_hier = 0
        for i in len(embd1):
            loss_hier+=loss_ftn(embd1[i],embd2[i])
        embd_loss = loss_hier/len(embd1)
        return embd_loss
    
    def forward(self, inp_bchwd: torch.Tensor, active_b1ff=None, vis=False, return_feat = False):
        # step1. Mask

        if active_b1ff.dtype == torch.bool:  # rand mask
            active_b1ff: torch.BoolTensor = self.mask(inp_bchwd.shape[0], inp_bchwd.device)  # (B, 1, f, f, f)
        else:
            pass


        encoder3D._cur_active = active_b1ff  # (b,1,7,7,8)
        active_b1hwd = active_b1ff.repeat_interleave(self.downsample_ratio, 2).repeat_interleave(self.downsample_ratio,
                                                                                                 3).repeat_interleave(
            self.downsample_ratio, 4)  # (B, 1, H, W)

        
        masked_bchwd = inp_bchwd * active_b1hwd # 둘다 ([4,1,112,112,128])

        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        fea_bcffs: List[torch.Tensor] = self.sparse_encoder(masked_bchwd)
        fea_bcffs.reverse()  # after reversion: from the smallest feature map to the largest

        # step3. Densify: get hierarchical dense features for decoding
        cur_active = active_b1ff  # (B, 1, f, f)
        to_dec = []
        # loss_pred = []
        if active_b1ff.dtype == torch.bool:
            for i, bcff in enumerate(fea_bcffs):  # from the smallest feature map to the largest
                if bcff is not None:
                    bcff = self.densify_norms[i](bcff)
                    mask_tokens = self.mask_tokens[i].expand_as(bcff)
                    bcff = torch.where(cur_active.expand_as(bcff), bcff,
                                    mask_tokens)  # fill in empty (non-active) positions with [mask] tokens
                    bcff: torch.Tensor = self.densify_projs[i](bcff)
                to_dec.append(bcff)
                # loss_pred.append(bcff)
                cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).repeat_interleave(2,
                                                                                                              dim=4)  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)
        else:
            for i, bcff in enumerate(fea_bcffs):  # from the smallest feature map to the largest
                if bcff is not None:
                    bcff = self.densify_norms[i](bcff)
                    bcff = bcff*cur_active.expand_as(bcff)
                    bcff: torch.Tensor = self.densify_projs[i](bcff)
                to_dec.append(bcff)
                # loss_pred.append(bcff)
                cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3).repeat_interleave(2,
                                                                                                              dim=4)  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)                                                                                                                                                                                                         dim=4)  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)
        # step4. Decode and reconstruct
        rec_bchwd = self.dense_decoder(to_dec)
        # rec_bchwd = 4,1,112,112,128
        if return_feat:
            return self.patchify(inp_bchwd), self.patchify(rec_bchwd), to_dec[0].flatten(start_dim=2).permute(0, 2, 1)

        else:
            inp, rec = self.patchify(inp_bchwd), self.patchify(
                rec_bchwd)  # inp and rec: (B, L = f*f*f, N = C*downsample_raito**3)

        if vis:
            mean = inp.mean(dim=-1, keepdim=True)
            var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .5
            masked_bchwd = inp_bchwd * active_b1hwd
            rec_bchwd = self.unpatchify(rec * var + mean)
            rec_or_inp = torch.where(active_b1hwd, inp_bchwd, rec_bchwd)
            return inp_bchwd, masked_bchwd, rec_or_inp

        else:
            return inp, rec

    def forward_loss(self, inp, rec, active_b1ff):

        mean = inp.mean(dim=-1, keepdim=True)
        var = inp.var(dim=-1, keepdim=True)
        inp = (inp - mean) / (var + 1.e-6) ** .5  # (B, L, C)

        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)  # (B, L, C) ==mean==> (B, L)
        non_active = active_b1ff.logical_not().int().view(active_b1ff.shape[0], -1)  # (B, 1, f, f) => (B, L)
        rec_loss = l2_loss * non_active
        recon_loss = l2_loss.mul_(non_active).sum() / (
                    non_active.sum() + 1e-8)  # loss only on masked (non-active) patches

        return recon_loss, rec_loss


    def forward_learning_loss(self, loss_pred, loss_target):
        """
        loss_pred: [N, L, 1]
        loss_target: [N, L]
        """
        # N, L = loss_target.shape
        # loss_pred = loss_pred[mask].reshape(N, L)

        # normalize by each image
        mean = loss_target.mean(dim=1, keepdim=True)
        var = loss_target.var(dim=1, keepdim=True)
        loss_target = (loss_target - mean) / (var + 1.e-6) ** .5  # [N, L, 1]

        loss = (loss_pred - loss_target) ** 2
        loss = loss.mean()
        return loss

    def patchify(self, bchwd):
        p = self.downsample_ratio
        h, w, d = self.fmap_h, self.fmap_w, self.fmap_d
        B, C = bchwd.shape[:2]
        bchwd = bchwd.reshape(shape=(B, C, h, p, w, p, d, p))
        bchwd = torch.einsum('bchpwqdg->bhwdpqgc', bchwd)
        bln = bchwd.reshape(shape=(B, h * w * d, C * p ** 3))  # (B, f*f*f, downsample_raito**3)
        return bln

    def unpatchify(self, bln):
        p = self.downsample_ratio
        h, w, d = self.fmap_h, self.fmap_w, self.fmap_d
        B, C = bln.shape[0], bln.shape[-1] // p ** 3
        bln = bln.reshape(shape=(B, h, w, d, p, p, p, C))
        bln = torch.einsum('bhwdpqgc->bchpwqdg', bln)
        bchwd = bln.reshape(shape=(B, C, h * p, w * p, d*p))
        return bchwd

    def __repr__(self):
        return (
            f'\n'
            f'[SparK.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[SparK.structure]: {super(SparK, self).__repr__().replace(SparK.__name__, "")}'
        )

    def get_config(self):
        return {
            # self
            'mask_ratio': self.mask_ratio,
            'densify_norm_str': self.densify_norm_str,
            'sbn': self.sbn, 'hierarchy': self.hierarchy,

            # enc
            'sparse_encoder.input_size': self.sparse_encoder.input_size,
            # dec
            'dense_decoder.width': self.dense_decoder.width,
        }

    def state_dict(self, destination=None, prefix='', keep_vars=False, with_config=False):
        state = super(SparK, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if with_config:
            state['config'] = self.get_config()
        return state

    def load_state_dict(self, state_dict, strict=True):
        config: dict = state_dict.pop('config', None)
        incompatible_keys = super(SparK, self).load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v = config.get(k, None)
                if ckpt_v != v:
                    err = f'[SparseMIM.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={ckpt_v})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err, file=sys.stderr)
        return incompatible_keys