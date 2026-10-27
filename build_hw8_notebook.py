"""Build the self-contained notebook from the implementation and report."""
from pathlib import Path
import nbformat as nbf

source = Path('hw8.py').read_text()
def section(start, end):
    return source[source.index(start):source.index(end)].strip()

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
nb.cells = [
    md('# Homework 8: Mini Diffusion Model\n\nMNIST, 100 diffusion steps, and one noise predictor shared by DDPM and DDIM.'),
    md('## Setup\n\nImages are scaled to [-1, 1]. Training uses all 60,000 MNIST training images, batch size 128, and 50 epochs. The random seed is 6812. Install the packages in `requirements.txt` before running this notebook.'),
    code(section('from pathlib', 'def forward_plot')),
    md(r'''## (a) Forward diffusion

Let $\alpha_t=1-\beta_t$ and $\bar\alpha_t=\prod_{s=1}^t\alpha_s$. The schedule increases linearly from 0.0001 to 0.02 over 100 steps. I sample a noisy image directly using

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad \epsilon\sim\mathcal N(0,I).$$

Index 0 represents the clean image, so $\bar\alpha_0=1$. The plot reuses one noise draw to make the changing noise level easier to see; it is not a simulated Markov trajectory.'''),
    code(section('def forward_plot', 'def train')),
    code("seed_all()\ndataset = MNIST('hw8_data', train=True, download=True)\nforward_plot(dataset.data)\nfrom IPython.display import display, Image\ndisplay(Image(filename=str(OUT / 'forward_diffusion.png')))\nprint('Final alpha_bar:', schedule('cpu')[2][-1].item())"),
    md(r'''## (b) Training the noise predictor

The CNN has eight convolutional layers with SiLU activations. A learned timestep embedding is added to the feature maps. It reduces the spatial resolution from 28×28 to 14×14, then upsamples and combines the features with a skip connection. Its output has shape `(batch, 1, 28, 28)`, matching the noise target.

Each training batch gets new Gaussian noise and independently sampled timesteps from 1 through 100. Adam minimizes $\mathbb E[\|\epsilon-\epsilon_\theta(x_t,t)\|^2]$, averaged over the pixels and batch. The plotted epoch loss is weighted by batch size.

The training cell resumes an existing checkpoint. In a fresh directory it trains all 50 epochs; remove `hw8_outputs/checkpoint.pt` to start over.'''),
    code(section('def train', '@torch.no_grad()\ndef ddpm')),
    code("model, history = train(epochs=50)\ndisplay(Image(filename=str(OUT / 'training_loss.png')))\nprint(f'Epochs: {len(history)}, first MSE: {history[0]:.6f}, last MSE: {history[-1]:.6f}')"),
    md(r'''## (c) DDPM sampling

Starting at $x_{100}\sim\mathcal N(0,I)$, the reverse mean is

$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right).$$

I use the fixed posterior variance $\tilde\beta_t=\beta_t(1-\bar\alpha_{t-1})/(1-\bar\alpha_t)$ and set $x_{t-1}=\mu_\theta+\sqrt{\tilde\beta_t}z$. Fresh Gaussian noise is added for $t>1$. The last step has no added noise.'''),
    code(section('@torch.no_grad()\ndef ddpm', '@torch.no_grad()\ndef ddim')),
    md(r'''## (d) DDIM sampling

DDIM uses the same trained network. For a selected timestep $t$ and the next smaller timestep $s$, I estimate

$$\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}},\qquad x_s=\sqrt{\bar\alpha_s}\hat x_0+\sqrt{1-\bar\alpha_s}\epsilon_\theta(x_t,t).$$

This is deterministic DDIM ($\eta=0$). The selected timesteps are rounded, evenly spaced points from 100 to 1; the final update goes to 0. The 50-step and 20-step versions therefore make exactly 50 and 20 network calls. No retraining is needed.

Each sampler starts with the same batch of 32 Gaussian images. Values are clipped only for display, not during the reverse process. Timings cover sampling on the reported device and exclude plotting.'''),
    code(section('@torch.no_grad()\ndef ddim', 'if __name__')),
    code("metrics = compare(model, history)\ndisplay(Image(filename=str(OUT / 'sampling_comparison.png')))"),
]
report = Path('HW8.md')
if report.exists():
    text = report.read_text(encoding='utf-8')
    nb.cells.append(md(text[text.index('## Discussion'):]))
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python'}}
nbf.write(nb, 'HW8.ipynb')
print('Created HW8.ipynb')
