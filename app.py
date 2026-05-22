"""
FusionCropNet V6 — Hugging Face Spaces Deployment
Multi-model crop classification with Gradio UI.
"""
import gradio as gr
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODELS = {}
CROP = {0:'wheat',1:'corn',2:'rice',3:'soybean',4:'cotton',5:'vegetable',6:'other'}
COLORS = ['#F4A460','#FFD700','#7CFC00','#228B22','#FFE4B5','#9370DB','#808080']

def get_model(name):
    if name not in _MODELS:
        from models.fusion_net_v6 import FusionCropNetV6
        from models.fusion_net_v5_edl import FusionCropNetV5EDL
        from models.fusion_net_v5pro import FusionCropNetV5Pro
        from models.fusion_net_v5 import FusionCropNetV5
        if 'V6' in name:
            m = FusionCropNetV6(opt_ch=10,sar_ch=5,dem_ch_in=5,num_classes=7,
                feat_dim=512,backbone='resnet18',pretrained=False,n_heads=4,n_layers=2)
        elif 'V5Pro' in name:
            m = FusionCropNetV5Pro(opt_ch=10,sar_ch=5,dem_ch_in=5,num_classes=7,
                feat_dim=512,backbone='resnet18',pretrained=False,n_heads=4,n_layers=2,
                use_carafe=False,dynamic_dropout=False,adaptive_kl=False)
        elif 'EDL' in name:
            m = FusionCropNetV5EDL(opt_ch=10,sar_ch=5,dem_ch_in=5,num_classes=7,
                feat_dim=512,backbone='resnet18',pretrained=False,n_heads=4,n_layers=2,
                use_v6_enhancements=False)
        else:
            m = FusionCropNetV5(opt_ch=10,sar_ch=5,dem_ch_in=5,num_classes=7,
                feat_dim=512,backbone='resnet18',pretrained=False,n_heads=4,n_layers=2)
        _MODELS[name] = m.to(DEVICE).eval()
    return _MODELS[name]

def classify(model_name, opt_f, sar_f, dem_f, size):
    import time; t0 = time.time()
    H = W = int(size); T = 12

    def load_or_rand(f, c, h, w, t=T):
        if f is not None:
            d = torch.from_numpy(np.load(f.name)).float()
            if d.dim() >= 2 and d.shape[0] > t: d = d[:t]
            if d.dim() >= 3: d = d[..., :h, :w]
            return d
        sh = (t, c, h, w) if c > 1 else (c, h, w)
        return torch.randn(*sh)

    opt = load_or_rand(opt_f, 10, H, W).unsqueeze(0).to(DEVICE)
    sar = load_or_rand(sar_f, 5, H, W).unsqueeze(0).to(DEVICE)
    dem = load_or_rand(dem_f, 5, H, W).unsqueeze(0).to(DEVICE)
    doy = torch.linspace(0,1,opt.shape[1]).unsqueeze(0).to(DEVICE)

    model = get_model(model_name)
    with torch.no_grad():
        if 'V6' in model_name:
            alpha, aux = model(opt, sar, dem, doy)
            aux_txt = f"LAI: {aux['lai'].mean().item():.3f} | Growth Stage: {aux['growth'].argmax(dim=1).item()} | Boundary: {aux['boundary'].mean().item()*100:.1f}%"
        else:
            alpha = model(opt, sar, dem, doy)
            aux_txt = "N/A (V6 only)"

    probs = (alpha/alpha.sum(1,keepdim=True)).squeeze(0)
    pred = probs.argmax(0).cpu().numpy()
    elapsed = time.time() - t0

    dist = {}
    for k in range(7):
        p = round((pred==k).sum()/pred.size*100,1)
        if p>0: dist[CROP[k]] = p
    dominant = max(dist,key=dist.get) if dist else '—'

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5.5))
    ax1.imshow(pred, cmap='tab10', vmin=0, vmax=6)
    ax1.set_title(f'Predicted — {dominant} ({round(probs.max(0)[0].mean().item()*100,1)}%)', fontsize=13)
    ax1.axis('off')
    ax1.legend(handles=[Patch(color=COLORS[i],label=CROP[i]) for i in range(7)], loc='lower right',fontsize=8,ncol=2)
    names, vals = list(dist.keys()), list(dist.values())
    bar_colors = [COLORS[list(CROP.keys())[list(CROP.values()).index(n)]] if n in CROP.values() else '#666' for n in names]
    ax2.bar(names, vals, color=bar_colors)
    ax2.set_title(f'Distribution ({elapsed:.1f}s, {str(DEVICE)})', fontsize=13)
    ax2.set_ylabel('%'); ax2.tick_params(axis='x',rotation=30)
    for i,v in enumerate(vals): ax2.text(i,v+1,f'{v}%',ha='center',fontsize=9)
    plt.tight_layout()

    info = f"### {model_name}\n**Dominant:** {dominant} | **Confidence:** {round(probs.max(0)[0].mean().item()*100,1)}% | **Time:** {elapsed:.1f}s\n\n{aux_txt}"
    return fig, info, dist

with gr.Blocks(title="FusionCropNet V6 — Crop Classification", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌾 FusionCropNet V6 — Crop Classification
    **Multi-modal remote sensing crop mapping.** Upload `.npy` files or use synthetic data.
    """)
    with gr.Row():
        with gr.Column(scale=1):
            model = gr.Dropdown(['V5 (Standard)','V5EDL','V5Pro (Flagship)','V6 (Next-Gen)'], value='V6 (Next-Gen)', label='Model')
            opt = gr.File(label='Optical .npy (T×10×H×W)', file_types=['.npy'])
            sar = gr.File(label='SAR .npy (T×5×H×W)', file_types=['.npy'])
            dem = gr.File(label='DEM .npy (5×H×W)', file_types=['.npy'])
            sz = gr.Slider(32,256,128,step=32,label='Image Size')
            run = gr.Button('🚀 Run Classification', variant='primary')
            synth = gr.Button('🎲 Use Synthetic Data')
        with gr.Column(scale=2):
            plot = gr.Plot(label='Output')
            info = gr.Markdown('Select model and click **Run**.')
            dist = gr.JSON(label='Distribution')

    synth.click(lambda: (None,None,None,'Ready — click Run'), outputs=[opt,sar,dem,info])
    run.click(classify, inputs=[model,opt,sar,dem,sz], outputs=[plot,info,dist])

    gr.Markdown("---\n### Models: V5 → V5EDL (+uncertainty) → V5Pro (+MIL/CARAFE) → **V6** (+TemporalLite, 5-path DEM, multi-scale Xattn, 5 aux heads)")

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)
