import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class DualModalPrototypeContrast(nn.Module):

    def __init__(self, pref_dim, emb_dim, hidden_dim=256,
                 temperature=0.2, logvar_min=-6.0, logvar_max=2.0, n_prototypes=4):
        super().__init__()
        self.K = n_prototypes
        self.temperature = temperature
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        # ── Per-modality user encoders (each takes its own 64-dim slice) ──
        self.cap_user_mu_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, emb_dim)
        )
        self.cap_user_logvar_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, emb_dim)
        )
        self.txt_user_mu_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, emb_dim)
        )
        self.txt_user_logvar_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, emb_dim)
        )

        # ── Per-modality prototypes (orthogonal init) ──
        if n_prototypes <= emb_dim:
            Q, _ = torch.linalg.qr(torch.randn(emb_dim, n_prototypes))
            proto_init = Q.T
        else:
            proto_init = torch.randn(n_prototypes, emb_dim) * 0.1

        # Caption prototypes
        self.cap_proto_mu = nn.Parameter(proto_init.clone())
        self.cap_proto_logvar = nn.Parameter(torch.zeros(n_prototypes, emb_dim))
        # Text prototypes
        self.txt_proto_mu = nn.Parameter(proto_init.clone())
        self.txt_proto_logvar = nn.Parameter(torch.zeros(n_prototypes, emb_dim))

        # ── Per-modality mixture projections (user_slice ⊕ pref_emb for richer signal) ──
        self.cap_mixture_proj = nn.Sequential(
            nn.Linear(emb_dim + pref_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_prototypes)
        )
        self.txt_mixture_proj = nn.Sequential(
            nn.Linear(emb_dim + pref_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, n_prototypes)
        )

        # ── Per-modality preference encoders ──
        self.cap_pref_mu_proj = nn.Sequential(
            nn.Linear(pref_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, emb_dim)
        )
        self.cap_pref_logvar_proj = nn.Sequential(
            nn.Linear(pref_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, emb_dim)
        )
        self.txt_pref_mu_proj = nn.Sequential(
            nn.Linear(pref_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, emb_dim)
        )
        self.txt_pref_logvar_proj = nn.Sequential(
            nn.Linear(pref_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, emb_dim)
        )

        # ── Per-modality blend factors ──
        self.cap_alpha = nn.Parameter(torch.tensor(0.5))
        self.txt_alpha = nn.Parameter(torch.tensor(0.5))

        # ── Shared per-term weights ──
        self.log_w_mu = nn.Parameter(torch.tensor(0.0))
        self.log_w_std = nn.Parameter(torch.tensor(0.0))

        self.emb_dim = emb_dim  # stored for slicing

    def _compute_contrast(self, user_mod_emb, pref_emb,
                          user_mu_proj, user_logvar_proj,
                          pref_mu_proj, pref_logvar_proj,
                          proto_mu, proto_logvar, mixture_proj, alpha_param):
        
        B = user_mod_emb.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=user_mod_emb.device)

        w_mu = torch.exp(self.log_w_mu)
        w_std = torch.exp(self.log_w_std)

        # ── User Gaussian (from modality-specific slice) ──
        u_mu = F.normalize(user_mu_proj(user_mod_emb), dim=1)
        u_lv = torch.clamp(user_logvar_proj(user_mod_emb), self.logvar_min, self.logvar_max)
        u_std = torch.exp(0.5 * u_lv)

        # ── Preference Gaussian ──
        p_mu = F.normalize(pref_mu_proj(pref_emb), dim=1)
        p_lv = torch.clamp(pref_logvar_proj(pref_emb), self.logvar_min, self.logvar_max)
        p_std = torch.exp(0.5 * p_lv)

        # ── User ↔ Preference ──
        d_user = w_mu * torch.cdist(u_mu, p_mu, p=2).pow(2) + w_std * torch.cdist(u_std, p_std, p=2).pow(2)

        # ── Prototype ↔ Preference ──
        # mixture input: user slice ⊕ pref emb for richer prototype assignment
        pi = F.softmax(mixture_proj(torch.cat([user_mod_emb, pref_emb], dim=1)), dim=1)
        k_mu = F.normalize(proto_mu, dim=1)
        k_lv = torch.clamp(proto_logvar, self.logvar_min, self.logvar_max)
        k_std = torch.exp(0.5 * k_lv)

        d_proto_mu = (k_mu.unsqueeze(0) - p_mu.unsqueeze(1)).pow(2).sum(dim=2)
        d_proto_std = (k_std.unsqueeze(0) - p_std.unsqueeze(1)).pow(2).sum(dim=2)
        d_proto = w_mu * d_proto_mu + w_std * d_proto_std
        d_proto_pair = torch.matmul(pi, d_proto.T)

        # ── Blend ──
        alpha = torch.sigmoid(alpha_param)
        dist = alpha * d_user + (1.0 - alpha) * d_proto_pair

        # ── Uncertainty weighting + contrast ──
        u_conf = torch.exp(-u_lv.mean(dim=1))
        p_conf = torch.exp(-p_lv.mean(dim=1))
        sw = torch.sqrt(u_conf * p_conf)
        sw = sw / (sw.sum() + 1e-8) * B

        logits = -dist / self.temperature
        labels = torch.arange(B, device=user_mod_emb.device)
        return (F.nll_loss(F.log_softmax(logits, dim=1), labels, reduction='none') * sw).mean()

    def forward(self, user_emb, pref_cap_emb, pref_txt_emb):
        # user_emb: [B, 192] = [user_image(64) | user_text(64) | user_caption(64)]
        u_txt = user_emb[:, self.emb_dim:2*self.emb_dim]        # user_text slice
        u_cap = user_emb[:, 2*self.emb_dim:3*self.emb_dim]      # user_caption slice

        loss_cap = self._compute_contrast(
            u_cap, pref_cap_emb,
            self.cap_user_mu_proj, self.cap_user_logvar_proj,
            self.cap_pref_mu_proj, self.cap_pref_logvar_proj,
            self.cap_proto_mu, self.cap_proto_logvar,
            self.cap_mixture_proj, self.cap_alpha)
        loss_txt = self._compute_contrast(
            u_txt, pref_txt_emb,
            self.txt_user_mu_proj, self.txt_user_logvar_proj,
            self.txt_pref_mu_proj, self.txt_pref_logvar_proj,
            self.txt_proto_mu, self.txt_proto_logvar,
            self.txt_mixture_proj, self.txt_alpha)
        return loss_cap, loss_txt


class SPAM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SPAM, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.retain_edge = config['retain_edge']
        self.mm_image_weight = config['mm_image_weight']
        self.mm_caption_weight = config['mm_caption_weight']

        # Hyperparameters
        self.proto_weight = self._cfg(config, 'proto_weight', 0.001)
        self.proto_temp = self._cfg(config, 'proto_temp', 0.2)
        self.proto_hidden = self._cfg(config, 'proto_hidden', 256)
        self.logvar_min = self._cfg(config, 'logvar_min', -6.0)
        self.logvar_max = self._cfg(config, 'logvar_max', 2.0)
        self.n_prototypes = self._cfg(config, 'n_prototypes', 4)

        self.pref_cap_file = self._cfg(config, 'user_pref_caption_file', 'user_pref_caption.npy')
        self.pref_txt_file = self._cfg(config, 'user_pref_text_file', 'user_pref_text.npy')

        self.n_nodes = self.n_users + self.n_items

        # ── Interaction graph ──
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self._build_norm_adj().to(self.device)
        self.edge_indices, self.edge_values = self._get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values = self.edge_values.to(self.device)

        # ── User embeddings ──
        self.user_image = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_text = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_caption = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_image.weight)
        nn.init.xavier_uniform_(self.user_text.weight)
        nn.init.xavier_uniform_(self.user_caption.weight)

        # ── Item features ──
        self.mm_adj = None
        self._build_modality_graph()

        # ── Dual-modal prototype contrast ──
        data_dir = os.path.abspath(config['data_path'] + config['dataset'])
        pref_cap, pref_txt = self._load_dual_pref(data_dir)
        if pref_cap is not None and pref_txt is not None:
            self.register_buffer('pref_cap_tensor', torch.FloatTensor(pref_cap))
            self.register_buffer('pref_txt_tensor', torch.FloatTensor(pref_txt))
            self.proto_contrast = DualModalPrototypeContrast(
                pref_dim=pref_cap.shape[1],
                emb_dim=self.embedding_dim,
                hidden_dim=self.proto_hidden,
                temperature=self.proto_temp,
                logvar_min=self.logvar_min,
                logvar_max=self.logvar_max,
                n_prototypes=self.n_prototypes
            )
        else:
            self.pref_cap_tensor = None
            self.pref_txt_tensor = None
            self.proto_contrast = None

        # ===== PCSA: Prototype-conditioned Semantic Anchoring =====
        self.use_pcsa = self._cfg(config, 'use_pcsa', True)          # ablation switch
        self.proto_anchor_tau = self._cfg(config, 'proto_anchor_tau', 0.2)
        self.lambda_div = self._cfg(config, 'lambda_div', 1e-4)

        # Project prototypes (embedding_dim) → item feature space (feat_embed_dim)
        self.pcsa_txt_proj = nn.Linear(self.embedding_dim, self.feat_embed_dim)
        self.pcsa_cap_proj = nn.Linear(self.embedding_dim, self.feat_embed_dim)

        # Learnable enhancement coefficients (sigmoid-constrained)
        self.anchor_beta_txt = nn.Parameter(torch.tensor(0.1))
        self.anchor_beta_cap = nn.Parameter(torch.tensor(0.1))


    @staticmethod
    def _cfg(config, key, default):
        v = config[key]
        if v is None: return default
        if isinstance(v, list): return v[0] if v else default
        return v

    def _load_dual_pref(self, data_dir):
        cap_path = os.path.join(data_dir, self.pref_cap_file)
        txt_path = os.path.join(data_dir, self.pref_txt_file)
        if not os.path.exists(cap_path) or not os.path.exists(txt_path):
            print(f'Warning: dual-pref files missing ({cap_path}, {txt_path}). Proto disabled.')
            return None, None
        cap = np.load(cap_path)[:self.n_users].astype(np.float32)
        txt = np.load(txt_path)[:self.n_users].astype(np.float32)
        for arr in [cap, txt]:
            n = np.linalg.norm(arr, axis=1, keepdims=True)
            n[n == 0] = 1.0
            arr[:] = arr / n
        return cap, txt


    def _build_modality_graph(self):
        if self.v_feat is None or self.t_feat is None or self.c_feat is None:
            return
        if self.v_feat is not None:
            self.image_emb = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            _, img_a = self._knn_adj(self.image_emb.weight.detach())
            self.mm_adj = img_a
        if self.t_feat is not None:
            self.text_emb = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            _, txt_a = self._knn_adj(self.text_emb.weight.detach())
            self.mm_adj = txt_a
        if self.c_feat is not None:
            self.cap_emb = nn.Embedding.from_pretrained(self.c_feat, freeze=False)
            self.cap_trs = nn.Linear(self.c_feat.shape[1], self.feat_embed_dim)
            _, cap_a = self._knn_adj(self.cap_emb.weight.detach())
            self.mm_adj = cap_a
        self.mm_adj = (self.mm_image_weight * img_a + (1 - self.mm_image_weight) * txt_a + self.mm_caption_weight * cap_a)
        del img_a, txt_a, cap_a

    def _knn_adj(self, emb):
        e = emb / (torch.norm(emb, p=2, dim=1, keepdim=True) + 1e-8)
        s = torch.mm(e, e.T)
        _, idx = torch.topk(s, self.knn_k, dim=1)
        sz = s.size(); del s
        i0 = torch.arange(idx.shape[0], device=self.device).unsqueeze(1).expand(-1, self.knn_k)
        ind = torch.stack((i0.flatten(), idx.flatten()), 0)
        return ind, self._laplacian(ind, sz)

    def _laplacian(self, ind, sz):
        a = torch.sparse.FloatTensor(ind, torch.ones_like(ind[0]), sz)
        rs = 1e-7 + torch.sparse.sum(a, -1).to_dense()
        r = torch.pow(rs, -0.5)
        return torch.sparse.FloatTensor(ind, r[ind[0]] * r[ind[1]], sz)

    def _build_norm_adj(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        M = self.interaction_matrix; Mt = M.transpose()
        A._update(dict(zip(zip(M.row, M.col + self.n_users), [1.] * M.nnz)))
        A._update(dict(zip(zip(Mt.row + self.n_users, Mt.col), [1.] * Mt.nnz)))
        s = np.array((A > 0).sum(axis=1).flatten())[0] + 1e-7
        L = sp.diags(np.power(s, -0.5)) * A * sp.diags(np.power(s, -0.5))
        L = sp.coo_matrix(L)
        return torch.sparse.FloatTensor(torch.LongTensor([L.row, L.col]), torch.FloatTensor(L.data), torch.Size(L.shape))

    def pre_epoch_processing(self):
        n = int(self.edge_values.size(0) * self.retain_edge)
        idx = torch.multinomial(self.edge_values, n)
        keep = self.edge_indices[:, idx]
        v = self._edge_norm(keep, (self.n_users, self.n_items))
        av = torch.cat((v, v))
        keep[1] += self.n_users
        ai = torch.cat((keep, keep.flip([0])), 1)
        self.sub_graph = torch.sparse.FloatTensor(ai, av, self.norm_adj.shape).to(self.device)

    def _edge_norm(self, idx, sz):
        a = torch.sparse.FloatTensor(idx, torch.ones_like(idx[0]), sz)
        r = 1e-7 + torch.sparse.sum(a, -1).to_dense()
        c = 1e-7 + torch.sparse.sum(a.t(), -1).to_dense()
        return torch.pow(r, -0.5)[idx[0]] * torch.pow(c, -0.5)[idx[1]]

    def _get_edge_info(self):
        r = torch.from_numpy(self.interaction_matrix.row)
        c = torch.from_numpy(self.interaction_matrix.col)
        e = torch.stack([r, c]).type(torch.LongTensor)
        return e, self._edge_norm(e, (self.n_users, self.n_items))



    def forward(self, adj):
        if self.v_feat is None or self.t_feat is None or self.c_feat is None:
            raise ValueError("SPCA4 needs image, text and caption features.")

        img_f = F.normalize(self.image_trs(self.image_emb.weight), dim=1)
        txt_f = F.normalize(self.text_trs(self.text_emb.weight), dim=1)
        cap_f = F.normalize(self.cap_trs(self.cap_emb.weight), dim=1)

        # ===== PCSA: Prototype-conditioned Semantic Anchoring =====
        if self.use_pcsa and self.proto_contrast is not None:
            # Project prototypes to item feature space
            txt_proto = F.normalize(self.pcsa_txt_proj(self.proto_contrast.txt_proto_mu), dim=1)   # [K, feat_dim]
            cap_proto = F.normalize(self.pcsa_cap_proj(self.proto_contrast.cap_proto_mu), dim=1)   # [K, feat_dim]

            # Item → Prototype Assignment
            S_txt = F.softmax(torch.mm(txt_f, txt_proto.T) / self.proto_anchor_tau, dim=1)                                 # [N_item, K]
            S_cap = F.softmax(torch.mm(cap_f, cap_proto.T) / self.proto_anchor_tau, dim=1)                                 # [N_item, K]

            # Build Semantic Anchor
            Anchor_txt = torch.mm(S_txt, txt_proto)                             # [N_item, feat_dim]
            Anchor_cap = torch.mm(S_cap, cap_proto)                             # [N_item, feat_dim]

            # Feature Enhancement with sigmoid-constrained coefficients
            txt_feat_enhanced = F.normalize(txt_f + torch.sigmoid(self.anchor_beta_txt) * Anchor_txt, dim=1)
            cap_feat_enhanced = F.normalize(cap_f + torch.sigmoid(self.anchor_beta_cap) * Anchor_cap, dim=1)
        else:
            txt_feat_enhanced = txt_f
            cap_feat_enhanced = cap_f

        u_emb = torch.cat([self.user_image.weight, self.user_text.weight, self.user_caption.weight], dim=1)
        i_emb = torch.cat([img_f, txt_feat_enhanced, cap_feat_enhanced], dim=1)

        # Modality propagation
        h = i_emb
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        # UI propagation
        ego = torch.cat((u_emb, i_emb), dim=0)
        out = [ego]
        for i in range(self.n_ui_layers):
            side = torch.sparse.mm(adj, ego)
            ego = side
            out += [ego]
        out = torch.stack(out, dim=1)
        out = out.mean(dim=1, keepdim=False)
        
        u_g, i_g = torch.split(out, [self.n_users, self.n_items], dim=0)
        return u_g, i_g + h



    def _bpr(self, u, pos, neg):
        return -torch.mean(F.logsigmoid(
            torch.sum(u * pos, dim=1) - torch.sum(u * neg, dim=1)))

    # ===== PCSA: Prototype Diversity Regularization =====
    def _prototype_diversity_loss(self):
        """Encourage prototypes to be orthogonal / well-separated.

        L_div = || P_txt P_txt^T - I ||_F  +  || P_cap P_cap^T - I ||_F
        where P = normalize(proto_mu).
        """
        txt_proto = F.normalize(self.proto_contrast.txt_proto_mu, dim=1)   # [K, d]
        cap_proto = F.normalize(self.proto_contrast.cap_proto_mu, dim=1)   # [K, d]

        P_txt = torch.mm(txt_proto, txt_proto.T)                            # [K, K]
        P_cap = torch.mm(cap_proto, cap_proto.T)                            # [K, K]

        I = torch.eye(self.n_prototypes, device=txt_proto.device)
        L_div_txt = torch.norm(P_txt - I, p='fro')
        L_div_cap = torch.norm(P_cap - I, p='fro')

        return L_div_txt + L_div_cap

    def calculate_loss(self, interaction):
        users, pos, neg = interaction[0], interaction[1], interaction[2]

        ua, ia = self.forward(self.sub_graph)
        self.build_item_graph = False

        # ① BPR
        loss_bpr = self._bpr(ua[users], ia[pos], ia[neg])

        # ② Dual-modal prototype contrast
        loss_proto = torch.tensor(0., device=ua.device)
        if self.proto_contrast is not None and self.proto_weight > 0:
            uniq = torch.unique(users)
            if uniq.size(0) > 1:
                pref_cap = self.pref_cap_tensor[uniq].to(ua.device)
                pref_txt = self.pref_txt_tensor[uniq].to(ua.device)
                loss_cap, loss_txt = self.proto_contrast(ua[uniq], pref_cap, pref_txt)
                loss_proto = loss_cap + loss_txt

        # ③ Prototype Diversity Regularization
        loss_div = torch.tensor(0., device=ua.device)
        if self.proto_contrast is not None and self.lambda_div > 0:
            loss_div = self._prototype_diversity_loss()

        return loss_bpr + self.proto_weight * loss_proto + self.lambda_div * loss_div

    def full_sort_predict(self, interaction):
        ue, ie = self.forward(self.norm_adj)
        return torch.matmul(ue[interaction[0]], ie.transpose(0, 1))
