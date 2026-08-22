


p$$
\begin{aligned}
&\text{cleaned\_avg\_linear\_speed} = \min\left(1, \max\left(0, \frac{\bar{v}}{\max(0.1, v_{\max})}\right)\right) \\
&\text{speed\_variance} = \min\left(1, \max\left(0, 1 - \frac{\sigma_v}{0.8\bar{v} + 0.1}\right)\right) \\
&\text{angular\_zcr} = \min\left(1, \max\left(0, 1 - \frac{1}{0.3} \cdot \frac{N_{\text{zc}}(\theta)}{N}\right)\right) \\
&\text{mid\_stop\_ratio} = \min\left(1, \max\left(0, 1 - \frac{\sum_{i \in \text{mid}} \mathbb{I}[v_i < 0.1]}{N}\right)\right) \\
&\text{zcr\_angular\_vel} = \min\left(1, \max\left(0, 1 - \frac{1}{2} \cdot \frac{N_{\text{zc}}(\omega)}{T + \varepsilon}\right)\right) \\
&\text{progress\_consistency} = \min\left(1, \max\left(0, \frac{\| \mathbf{p}_N - \mathbf{p}_1 \|}{\sum_{i=1}^{N-1} \| \mathbf{p}_{i+1} - \mathbf{p}_i \| + \varepsilon}\right)\right) \\
&\text{angular\_smoothness} = \min\left(1, \max\left(0, 1 - \frac{1}{N} \sum_{j=1}^{N} \mathbb{I}\left[ |\Delta \theta_j| > \tau \right]\right)\right), \quad \tau = \text{P}_{75}(|\Delta \theta|) + 0.5 \cdot \text{std}(|\Delta \theta|) \\
&\text{area\_redundancy} = \min\left(1, \max\left(0, \frac{1}{0.25} \cdot \frac{\frac{1}{N} \sum_{i=1}^{N} \| \mathbf{p}_i - \bar{\mathbf{p}} \|}{\sum_{i=1}^{N-1} \| \mathbf{p}_{i+1} - \mathbf{p}_i \| + \varepsilon}\right)\right)
\end{aligned}
$$
