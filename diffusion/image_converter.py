class ImageConverter:
    def __init__(self, h, w, patch_size, vae):
        self.h = h
        self.w = w
        self.patch_size = patch_size
        if isinstance(vae, str):
            vae = AutoencoderKL.from_pretrained(vae).to(device)
        self.vae = vae
        self.num_patch = (h * w) // (patch_size * patch_size)
        self.patch_h = h // patch_size
        self.patch_w = w // patch_size

    def timeseq2patchseq(self, x):
        B, T, C, H, W = x.shape
        patch_size = self.patch_size
        assert H % patch_size == 0 and W % patch_size == 0
        h, w = H // patch_size, W // patch_size

        x = x.reshape(B, T, C, h, patch_size, w, patch_size)  # 分 patch
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()  # B, T, h, w, C, p, p
        x = x.reshape(B, T * h * w, C, patch_size, patch_size)  # B, T*Num_Patch, C, P, P
        return x

    def decode(self, seq):
        return self.vae.decode(seq.unsqueeze(0).to(seq.device)/0.18215).sample