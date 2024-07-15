# asset-pricing
Where we keep the code for our asset pricing research.

## Setup

To create the conda environment (and python interpreter),
please run the following command:
```bash
conda env create -f environment.yml
```

To update the existing conda environment, please run:
```bash
conda env update -f environment.yml
```

To format the entire code base, please run:
```bash
black -l 80 ./
```

## Transformer over the Asset Space

Here, I formally define how I thought we could use a
transformer neural network to improve Pelger's 
adversarial asset pricing model.

Let $I_{t,i} \in \mathbb{R}^{F}$ be the
information set at time $t$ for asset $i$ and let
$R_{t,i} \in \mathbb{R}$ denote the *excess return*
of that asset at time $t$ for $1 \leq i \leq N, 1 \leq t \leq T$.

> Pelger's model maps $I_{t,i}$ to $\omega_{t,i}$ (the SDF weight) *point-wise*.

This point-wise modeling can not capture 
*interactions within the asset space*.
What I mean by this is that the *unnormalized SDF weight* of
any asset only depends on its information set at that time.

**Idea**: We use a transformer neural network *at every time
step $t$ to map the sequence of information sets to a
sequence of SDF weights*.

Formally, the transformer neural network is a parametrized
mapping as follows:
$$
T_\theta : \mathbb{R}^{N \times F} \to \mathbb{R}^{N} \\
(I_{t,1}, I_{t,2}, \ldots, I_{t,N_t}) \mapsto (\omega_{t,1}, \omega_{t,2}, \ldots, \omega_{t,N_t})
$$
Note here that the number of assets at each point in time
$N_t$
is allowed to vary arbitrarily and *no positional encoding
should be used*.

#### Why no positional encoding?

We want the model to produce the same SDF weights
if the order of assets is changed (i.e. they are permuted).
In an example, this means that for some point in time $t$:
$$
\begin{align*}
    T_\theta (I_{t, 1}, I_{t, 2}) &= (\omega_{t, 1}, \omega_{t, 2}) \\
    (\omega_{t, 2}, \omega_{t, 1}) &= T_\theta (I_{t, 2}, I_{t, 1})
\end{align*}
$$
Essential, the transformer model represents a *set-to-set*
mapping rather than a *sequence-to-sequence* mapping.

#### What can this model capture that Pelger's model can't?

Mainly, the transformer can pick up on *interactions across the asset space*.
I.e. if many large-cap assets have lower $BE/ME$ ratios,
the model may assign small cap assets higher SDF weights.

Pelger's model could, theoretically capture things effects like
this but only through normalization of the SDF weights.
However, Pelger's point-wise mapping cannot model this 
interaction perfectly if events are not neatly correlated.
That is to say that Pelger's SDF could lower weights of
large-cap assets with low $BE/ME$ ratios but 
*it then relies on small cap assets to be presents* in
the asset space at that point in time.

Differently formulated, Pelger's model can only capture
interactions through the normalization of the raw
SDF weights.
There are, theoretically, more complex interactions that
cannot be captured this way.

#### An example of a complex interaction

In the training set, we have 2 small cap stocks and 1 large
cap stock (and only feature is $ME$).
Pelger's SDF learns to assign higher weights to small
cap stocks but only so much that the normalized
SDF weights form a good portfolio.

Let's say, for simplicity:
$$
\omega_{\text{small cap}} = 1, \qquad
\omega_{\text{large cap}} = 0.5
$$
We then yield the normalized SDF weights:
$(0.4, 0.4, 0.2)$.
$$\implies
\text{
    Roughly 80\% small cap in SDF portfolio
}
$$

Now, however in the test set we have 1 small cap stock
and 2 large cap stocks.
Since Pelger's model doesn't know the whole set of assets,
but only maps market-equity to SDF weight point-wise,
we yield (with the same unnormalized weights):
$(0.5, 0.25, 0.25)$
$$\implies \text{
    Roughly 50\% small cap in SDF portfolio
}
$$

Essentially the problem is that a point-wise model
assumes that the *relative distribution of assets*
stays the same as it cannot count the number of small/large
cap stocks in the asset space.
