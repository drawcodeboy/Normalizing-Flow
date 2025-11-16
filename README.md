# Normalizing Flow
* Implementation of <b><i><a href="https://arxiv.org/abs/1505.05770">Variational Inference with Normalizing Flows</a></i></b> with PyTorch
# Settings
```
conda create -n nf python=3.12
conda activate nf
pip install -r requirements.txt

python train.py --config=dlgm.nf10.mnist
python test.py --config=dlgm.nf10.mnist
```
# Results
<table align="center">
  <tr>
    <td align="center">
      <img src="assets/figure_4.jpg" width="500"><br>
      <em>(a) This repo's result (Flow length 0, 10, 20, 40)</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/figure_4_wo_weight_init.png" width="500"><br>
      <em>(b) This repo's result (w/o weight initialization, Flow length 0, 10, 20)</em>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/figure_4_original_paper.png" width="500"><br>
      <em>(c) Original paper</em>
    </td>
  </tr>
</table>

# Limitation & Report
* The performance difference from the original paper is shown in the Results section. In my implementation, the performance gain from adding Flow is smaller than what the original paper reports. To understand this discrepancy, I reviewed the paper again and identified one key difference: in the original implementation, the parameters of each Flow are generated from the output of the Inference Network <i>(in Sec 4.2.)</i>. Aside from this point, I am confident that the rest of my implementation follows the paper faithfully.
* This means the original paper uses flow-specific parameters conditioned on the encoder output, whereas my implementation uses separate learnable parameters for each Flow. While examining this detail, I made an interesting finding: <b> for Planar Flow, the choice of weight initialization has a critical impact when tanh is used as the nonlinearity. </b> The reasons are as follows.

    - The term $\mathbf{w}^\top\mathbf{z}$ always passes through the nonlinearity $h(\cdot)$ and its derivative $h'(\cdot)$ for both forward computation and the log-det Jacobian. With $h(\cdot)=\text{tanh}$, the useful gradient region is limited.
        
    - Assuming $b=0$, the value $\mathbf{w}^\top\mathbf{z}$ needs to lie approximately in the interval $[−2,2]$ to maintain meaningful gradients.
        * If $\mathbf{w}$ is too large, $\text{tanh}(\mathbf{w}^\top\mathbf{z})$ saturates and gradients vanish, making the flow impossible to train.
        * If $\mathbf{w}$ is too small, $h(\mathbf{w}^\top\mathbf{z})$ becomes close to zero, causing $f(\mathbf{z})\approx\mathbf{z}$. This reduces the flow to an identity mapping and effectively disables it.
        $$f(\mathbf{z})=\mathbf{z}+\mathbf{u}h(\mathbf{w}^\top\mathbf{z}+b)$$
    
    - The derivative $h'(\mathbf{w}^\top\mathbf{z})$ appears in the logdet-Jacobian:
        * If $\mathbf{w}$ is too large, the inner term of the log becomes close to 1, making the logdet-Jacobian close to 0.
        * If $\mathbf{w}$ is too small, then in
        $$\psi(\mathbf{z})=h'(\mathbf{w}^\top\mathbf{z}+b)\mathbf{w}$$
        * the derivative of tanh behaves close to 1 near 0, so the entire expression scales with $\mathbf{w}$. This again makes the log-det term close to 0.
        * A log-det Jacobian that is too small leads to extremely small gradient magnitudes, slowing or preventing learning:
        $$\frac{\partial}{\partial\mathbf{u}}\log\vert1+\mathbf{u}^\top\psi(\mathbf{z})\vert=\frac{\psi(\mathbf{z})}{1+0}=\psi(\mathbf{z})$$
        * When $\mathbf{w}$ is too small, $\psi(\mathbf{z})$ is small as well, causing the gradient to shrink.
* Given that each Flow has its own parameters, <a href="https://github.com/VincentStimper/normalizing-flows">a repository</a> with an implementation similar to mine initializes the weights using a scheme similar to Xavier initialization. I applied the same strategy, and the difference between the initialized model (a) and the non-initialized model (b) shows a clear performance gap. <b>This supports the finding that weight initialization plays a critical role in the effectiveness of Planar Flow.</b>
# References
```
@inproceedings{rezende2015variational,
    title={Variational inference with normalizing flows},
    author={Rezende, Danilo and Mohamed, Shakir},
    booktitle={International conference on machine learning},
    pages={1530--1538},
    year={2015},
    organization={PMLR}
}

@inproceedings{salakhutdinov2008quantitative,
    title={On the quantitative analysis of deep belief networks},
    author={Salakhutdinov, Ruslan and Murray, Iain},
    booktitle={Proceedings of the 25th international conference on Machine learning},
    pages={872--879},
    year={2008},
    organization={ACM}
}
```