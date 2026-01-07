"""
MNIST Unlearning Script with ODEBlock Injection.
Target: Unlearn specific classes by injecting a learnable ODE dynamics block.
Note: Modified to compute MMD & Leakage at Epoch 5.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import csv
import time
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 尝试导入 torchdiffeq，如果没有则报错
try:
    from torchdiffeq import odeint
except ImportError:
    print("Error: torchdiffeq not found. Please install it: pip install torchdiffeq")
    sys.exit(1)

# --- 1. 环境与路径设置 ---
try:
    project_root = "/path/....."
    if project_root not in sys.path:
        sys.path.append(project_root)

    from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
    from torchcfm.models.unet import UNetModel
    
    sys.path.append(os.path.join(project_root, "experiments/classifiers/mnist/"))
    from eval_classifiers import get_mnist_classifier
    
    # 自动选择 GPU
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        from utils.select_gpu import pick_gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu())
        
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


# ==========================================
# [辅助函数] MMD 计算工具
# ==========================================
def compute_mmd_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算高斯核 MMD"""
    n_samples = int(x.size(0))
    m_samples = int(y.size(0))
    x = x.view(n_samples, -1)
    y = y.view(m_samples, -1)
    total = torch.cat([x, y], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples + 1e-8)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / (bandwidth_temp + 1e-8)) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def calc_mmd_loss(x, y):
    if x.size(0) == 0 or y.size(0) == 0: return 0.0
    kernels = compute_mmd_kernel(x, y)
    n = x.size(0)
    xx = kernels[:n, :n]
    yy = kernels[n:, n:]
    xy = kernels[:n, n:]
    yx = kernels[n:, :n]
    loss = torch.mean(xx) + torch.mean(yy) - torch.mean(xy) - torch.mean(yx)
    return loss.item()

# ==========================================
# [核心组件] ODEBlock 定义 (保持不变)
# ==========================================

class UnlearningLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.GroupNorm(min(32, hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, 1, 1) # 必须映射回 in_channels
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

class ODEfunc(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.unlearn = UnlearningLayer(in_channels, hidden_channels)
        self.out = []
        self.nfe = 0

    def forward(self, t, x):
        self.out.append(x)
        self.nfe += 1
        return self.unlearn(x)

class ODEBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.odefunc = ODEfunc(in_channels, hidden_channels)
        self.rtol = 1e-4
        self.atol = 1e-3
        self.method = 'euler'    
        self.step_size = 0.4       
        self.perturb = False
        self.hiddenEmbed = None
        
    def forward(self, x):
        self.t = torch.tensor([0, 1.6]).float().type_as(x)      
        self.odefunc.out = []
        self.odefunc.nfe = 0
        out = odeint(self.odefunc, x, self.t, 
                     rtol=self.rtol, atol=self.atol,
                     method=self.method, 
                     options=dict(step_size=self.step_size, perturb=self.perturb))
        outs = self.odefunc.out
        outs.append(out[-1])
        h_out = torch.cat(outs, dim=1) 
        self.hiddenEmbed = h_out
        return out[-1]

# ==========================================
# [工具] 模型注入器 (保持不变)
# ==========================================

class ODEWrapper(nn.Module):
    def __init__(self, original_block, channels, hidden_dim=256):
        super().__init__()
        self.original_block = original_block
        self.ode_block = ODEBlock(channels, hidden_dim)

    def forward(self, x, emb=None, *args, **kwargs):
        h = self.original_block(x, emb, *args, **kwargs)
        h_ode = self.ode_block(h)
        return h_ode

def inject_ode_block(model):
    base_ch = model.model_channels
    ch_mult = model.channel_mult 
    mid_channels = base_ch * ch_mult[-1]
    print(f"Injecting ODEBlock into Middle Block (Channels: {mid_channels})...")
    model.middle_block = ODEWrapper(model.middle_block, mid_channels, hidden_dim=256)
    return model

# ==========================================
# 配置与训练器 (Train loop 微调)
# ==========================================

class Config:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = args.seed
        self.data_dir = Path(project_root) / "data"
        self.checkpoint_path = Path(args.checkpoint_path)

class UnlearningTrainer:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.device = config.device
        self._set_seed()
        self._setup_dirs()
        self._load_models()
        self._setup_data()

    def _set_seed(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        torch.backends.cudnn.deterministic = True

    def _setup_dirs(self):
        name_parts = ["mnist", "unlearn", f"forget{''.join(map(str, self.args.forget_labels))}"]
        if self.args.ode_block: name_parts.append("ODEBlock") 
        
        self.run_dir = Path(project_root) / "experiment_runs" / "unlearn_results" / "_".join(name_parts)
        self.models_dir = self.run_dir / "models"
        self.results_dir = self.run_dir / "results"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {self.run_dir}")

    def _create_unet(self):
        return UNetModel(
            dim=(3, 32, 32),
            num_channels=128,           
            num_res_blocks=2,           
            channel_mult=(1, 2, 2, 2), 
            class_cond=True,
            num_classes=11,
            attention_resolutions="16",
            num_heads=4,
            dropout=0.1
        ).to(self.device)

    def _load_models(self):
        print(f"Loading pre-trained model from: {self.config.checkpoint_path}")
        if not self.config.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {self.config.checkpoint_path}")
            
        self.train_model = self._create_unet()
        state_dict = torch.load(self.config.checkpoint_path, map_location=self.device)
        self.train_model.load_state_dict(state_dict)
        
        if self.args.ode_block:
            print("--- Enabling ODEBlock Mode ---")
            self.train_model = inject_ode_block(self.train_model)
            self.train_model.to(self.device)
            print("Freezing base UNet parameters...")
            for name, param in self.train_model.named_parameters():
                param.requires_grad = False
            print("Unfreezing ODEBlock parameters...")
            trainable_count = 0
            for name, param in self.train_model.named_parameters():
                if "ode_block" in name or "odefunc" in name: 
                    param.requires_grad = True
                    trainable_count += param.numel()
            print(f"Injection complete. Trainable params: {trainable_count}")
        else:
            print("--- Standard Fine-tuning Mode (No ODEBlock) ---")
        
        self.baseline_model = self._create_unet()
        self.baseline_model.load_state_dict(state_dict)
        self.baseline_model.eval()
        for p in self.baseline_model.parameters(): p.requires_grad = False
        
        self.classifier = get_mnist_classifier(self.device)
        self.classifier.eval()
        
        self.clf_normalizer = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], 
            std=[0.2023, 0.1994, 0.201]
        )

    def _setup_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.MNIST(self.config.data_dir, train=True, download=True, transform=transform)
        # Load Testset for Leakage check
        testset = datasets.MNIST(self.config.data_dir, train=False, download=True, transform=transform)
        
        self.forget_labels = self.args.forget_labels
        self.retain_labels = [i for i in range(10) if i not in self.forget_labels]
        
        forget_idx = [i for i, (_, y) in enumerate(trainset) if y in self.forget_labels]
        self.forget_loader = DataLoader(Subset(trainset, forget_idx), batch_size=self.args.batch, shuffle=True, num_workers=4, drop_last=True)
        
        retain_idx = [i for i, (_, y) in enumerate(trainset) if y not in self.forget_labels]
        self.retain_loader = DataLoader(Subset(trainset, retain_idx), batch_size=self.args.batch, shuffle=True, num_workers=4, drop_last=True)
        self.retain_iter = iter(self.retain_loader)
        
        # Test Loader for Forget Class (Leakage)
        f_test_idx = [i for i, (_, y) in enumerate(testset) if y in self.forget_labels]
        self.forget_test_loader = DataLoader(Subset(testset, f_test_idx), batch_size=self.args.batch, shuffle=False)

    def train(self):
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.train_model.parameters()), lr=self.args.lr)
        fm = ConditionalFlowMatcher(sigma=0.0)
        
        csv_path = self.results_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "total_loss", "forget_loss", "retain_loss", "retain_acc", "forget_acc", "time"])

        print(f"\n--- Starting Unlearning (ODEBlock: {self.args.ode_block}) ---")
        
        # 只跑 5 轮 (或 args.epochs)
        for epoch in range(1, self.args.epochs + 1):
            self.train_model.train()
            
            total_loss_epoch = 0.0
            l_f_sum, l_r_sum = 0.0, 0.0
            start_time = time.time()
            
            pbar = tqdm(self.forget_loader, desc=f"Epoch {epoch}")

            for xb_f, yb_f in pbar:
                xb_f, yb_f = xb_f.to(self.device), yb_f.to(self.device)
                
                # --- 1. Forget Loss ---
                x0_f = torch.randn_like(xb_f)
                target_noise = torch.randn_like(xb_f) 
                t_f, xt_f, ut_f = fm.sample_location_and_conditional_flow(x0_f, target_noise)
                vt_f = self.train_model(t_f, xt_f, yb_f)
                loss_forget = F.mse_loss(vt_f, ut_f)

                # --- 2. Retain Loss ---
                try:
                    xr_b, yr_b = next(self.retain_iter)
                except StopIteration:
                    self.retain_iter = iter(self.retain_loader)
                    xr_b, yr_b = next(self.retain_iter)
                xr_b, yr_b = xr_b.to(self.device), yr_b.to(self.device)
                
                x0_r = torch.randn_like(xr_b)
                t_r, xt_r, _ = fm.sample_location_and_conditional_flow(x0_r, xr_b)
                vt_r_student = self.train_model(t_r, xt_r, yr_b)
                with torch.no_grad():
                    vt_r_teacher = self.baseline_model(t_r, xt_r, yr_b)
                loss_retain = F.mse_loss(vt_r_student, vt_r_teacher)

                # --- Total Loss ---
                loss = (self.args.forget_weight * loss_forget) + \
                       (self.args.retain_weight * loss_retain)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss_epoch += loss.item()
                l_f_sum += loss_forget.item()
                l_r_sum += loss_retain.item()
                pbar.set_postfix(L_tot=loss.item(), L_f=loss_forget.item(), L_r=loss_retain.item())

            epoch_time = time.time() - start_time
            avg_loss = total_loss_epoch / len(self.forget_loader)
            
            print(f"Evaluating Epoch {epoch}...")
            r_acc, f_acc, gen_forget, gen_retain = self.evaluate(return_samples=(epoch == 5))
            
            print(f"Ep {epoch} | Loss: {avg_loss:.4f} | R-Acc: {r_acc:.2f}% | F-Acc: {f_acc:.2f}% | Time: {epoch_time:.1f}s")
            
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, avg_loss, l_f_sum, l_r_sum, r_acc, f_acc, epoch_time])
            
            # === [新增] 第 5 轮计算 MMD 和 Leakage ===
            if epoch == 5:
                print("\n=== Final Epoch 5 Check: MMD & Leakage ===")
                self.calc_final_metrics(gen_forget, gen_retain, fm)
                # 跑完 5 轮直接退出
                break

    def calc_final_metrics(self, gen_forget, gen_retain, fm):
        """仅在第5轮被调用的辅助函数，计算 MMD 和 Leakage"""
        # 1. 计算 MMD
        print("Computing MMD...")
        # 获取真实样本 batch
        real_forget, _ = next(iter(self.forget_loader))
        real_retain, _ = next(iter(self.retain_loader))
        real_forget = real_forget.to(self.device)[:len(gen_forget)]
        real_retain = real_retain.to(self.device)[:len(gen_retain)]
        
        gf = torch.cat(gen_forget) if gen_forget else torch.randn(1, 3, 32, 32, device=self.device)
        gr = torch.cat(gen_retain) if gen_retain else torch.randn(1, 3, 32, 32, device=self.device)
        
        # 确保尺寸匹配
        n = min(gf.size(0), real_forget.size(0))
        mmd_f = calc_mmd_loss(gf[:n], real_forget[:n])
        
        m = min(gr.size(0), real_retain.size(0))
        mmd_r = calc_mmd_loss(gr[:m], real_retain[:m])
        
        print(f"  >> MMD (Retain): {mmd_r:.4f} (Lower is better)")
        print(f"  >> MMD (Forget): {mmd_f:.4f} (Should be High)")
        
        # 2. 计算 Leakage (Loss Gap)
        print("Computing Leakage (MIA Gap)...")
        # 定义计算 Loss 的小函数
        def get_loss(loader):
            losses = []
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                x0 = torch.randn_like(x)
                t, xt, ut = fm.sample_location_and_conditional_flow(x0, x) # Target is REAL Image
                with torch.no_grad():
                    vt = self.train_model(t, xt, y)
                losses.append(F.mse_loss(vt, ut).item())
                if len(losses) > 5: break 
            return np.mean(losses)

        train_loss = get_loss(self.forget_loader)
        test_loss = get_loss(self.forget_test_loader)
        gap = test_loss - train_loss
        
        print(f"  >> Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print(f"  >> Leakage Gap: {gap:.4f} (Small/Negative is good)")


    @torch.no_grad()
    def evaluate(self, return_samples=False):
        self.train_model.eval()
        
        n_per_class = 50      
        steps = 50            
        guidance_scale = 2.0  
        
        t_span = torch.linspace(0, 1, steps, device=self.device)
        accs = {}
        
        # 用于收集样本
        collected_forget = []
        collected_retain = []

        def cfg_ode_func(t, x, y_cond, y_null, scale):
            t_batch = t.expand(x.shape[0])
            v_cond = self.train_model(t_batch, x, y_cond)
            v_null = self.train_model(t_batch, x, y_null)
            return v_null + scale * (v_cond - v_null)

        for lbl in range(10):
            y_cond = torch.full((n_per_class,), lbl, dtype=torch.long, device=self.device)
            y_null = torch.full((n_per_class,), 10, dtype=torch.long, device=self.device)
            x0 = torch.randn(n_per_class, 3, 32, 32, device=self.device)
            
            ode_func = lambda t, x: cfg_ode_func(t, x, y_cond, y_null, guidance_scale)
            
            traj = odeint(ode_func, x0, t_span, method='euler')
            samples = traj[-1]
            
            # 如果是第5轮，收集样本
            if return_samples:
                if lbl in self.forget_labels:
                    collected_forget.append(samples)
                elif len(collected_retain) < 1: # Retain只存一点点就够了
                    collected_retain.append(samples)
            
            samples_01 = torch.clamp((samples * 0.5) + 0.5, 0, 1)
            samples_clf = self.clf_normalizer(samples_01)
            
            preds = self.classifier(samples_clf).argmax(dim=1)
            acc = (preds == y_cond).sum().item() / n_per_class * 100
            accs[lbl] = acc
            
        r_acc = sum(accs[l] for l in self.retain_labels) / len(self.retain_labels)
        f_acc = sum(accs[l] for l in self.forget_labels) / len(self.forget_labels) if self.forget_labels else 0.0
        
        self.train_model.train()
        
        if return_samples:
            return r_acc, f_acc, collected_forget, collected_retain
        else:
            return r_acc, f_acc, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 请根据实际路径修改默认值
    parser.add_argument("--checkpoint_path", type=str, 
                        default="/path/...../checkpoint_ep100.pth", 
                        help="Path to pre-trained UNet checkpoint")
    parser.add_argument("--forget_labels", nargs="*", type=int, default=[0], help="Class to forget")
    parser.add_argument("--lr", type=float, default=1e-2) 
    parser.add_argument("--epochs", type=int, default=5) # 默认为 5 轮
    parser.add_argument("--batch", type=int, default=128) 
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ode_block", action="store_true", help="Enable ODEBlock injection and freezing")
    
    # 权重
    parser.add_argument("--forget_weight", type=float, default=1.0)
    parser.add_argument("--retain_weight", type=float, default=1.0)
    
    args = parser.parse_args()
    config = Config(args)
    trainer = UnlearningTrainer(config, args)

    trainer.train()
