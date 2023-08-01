---
layout: post
title: Variational Inference
---


**Variational inference** is a technique for approximating intractable posterior distributions in Bayesian inference. The basic idea is to construct a simpler, tractable distribution (the variational distribution) that approximates the true posterior distribution, and then optimize the parameters of this variational distribution to minimize the difference between the two distributions.


Mathematically, we can write the problem of variational inference as follows. Given observed data $$x$$ and a prior distribution $$p(z)$$ over latent variables $$z$$, we want to compute the posterior
distribution $$p(z|x)$$. However, this posterior is often intractable to compute directly. Instead, we introduce a family of variational distributions $$q(z|\theta)$$ parameterized by $$\theta$$, and minimize the **Kullback-Leibler (KL) divergence** between $$q(z|\theta)$$ and $$p(z|x)$$:


$$\min_{\theta} \mathrm{KL}(q(z|\theta) || p(z|x))$$

where 

$$\mathrm{KL}(q(z|\theta) || p(z|x)) = \int q(z|\theta) \log \frac{q(z|\theta)}{p(z|x)}dz$$

