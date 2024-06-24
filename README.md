# ScalMatErrorBriefWorkflow
<table>
<tr>
    <td><sub> First of all. I'm not a mathematician by any means so apologize if something is incorrect, wrong or not proper in the perspective of  math. This is  just the error analysis from the perspective of developers like me. 
Because last time I tried to figure out a math problem involving 200 candies I got nothing in my mind other than that guy who bought those candies must having diabetes
    </sup>
    </td>
  </tr>
</table>

<ins>A brief transformation of errors workflow in linear model (scalar & matrices perspective).</ins>

Linearity is pretty common model to encounter in the estimation realm. I mean.., it's like being everywhere. During the computation it's pretty common to use the matrix/vector formation but why is it computed in such way? Do we really understand it? Implementing/solving the model using a tool such as _Tensorflow_ tends to be _easy_ but most individuals don't really understand why exactly it's computed in such way.

This is the common model having the coefficients of $c_{1}$ for the _gradient_ & $c_{0}$ _intercept_

$t = f(x;c_{0},c_{1}) = c_{0}+c_{1}x$

and the lost would be described as 

$L_{n} = t_{n}- f(x;c_{0},c_{1})^2$ , $L_{n}>0$

but what is needed is the lost for all observed values ($N$ data points), hence (after adjustment to the $c_{0}$ & $c_{1}$) . We come the conclusion that we need is

```math
argmin_{c_{0},c_{1}}= \frac{1}{N} \sum_{n=1}^{N}{L}_{n}(t_{n}, f\left(x_{n} ; c_{0}, c_{1}\right)^2)
```
that yields 
```math
\frac{1}{N} \sum_{n=1}^{N}\left(c_{1}^{2} x_{n}^{2}+2 c_{1} c_{n}\left(c_{0}-t_{n}\right)+c_{0}^{2}-2 c_{0} t_{n}+t_{n}^{2}\right)
```

Now, the gradient _needs to be zero_ (with respect to $c$) by deriving it (the goal is the _minima_ isn't it?? ).


_I'm not a mathematician_ by any means but i think it's a valid point to mention the minima of it's function

```math
\frac{1}{N}\sum_{n=1}^{N}[c_{1}^{2}x_{n}^{2}+2c_{1}x_{n}c_{0}-2c_{1}x_{n}t_{n}]
```
---
<sup>
With the intention of simplication the equation has to shorten. The following are the steps taken


1. distribute that $1/N$ inside the summation

```math
\frac{1}{N} \sum_{n=1}^{N} c_{1}^{2} x_{n}^{2} + \frac{1}{N} \sum_{n=1}^{N} 2c_{1} x_{n} (c_{0} - t_{n})
```
2. And then, left side simplified into

   ```math
   \frac{c_{1}^{2}}{N}( \sum_{n=1}^{N} x_{n}^{2})
   ```
    and the right side simplified into

  ```math
  \frac{2c_{1}}{N} \left( \sum_{n=1}^{N} x_{n} (c_{0} - t_{n}) \right)
  ```
</sup>

---

Hence,

```math
c_{1}^2 1/N  (\sum_{n=1}^N x_{n}^2 ) + 2c_{1}1/N (\sum_{n=1}^N x_{n}(c_{0}-t_{n}))
```
partial derivative with respect to $c_{1}$ yields

```math
\partial L/ \partial c_{1}= 2c_{1}  1/N  (\sum_{n=1}^N x_{n}^2 ) + 2/N (\sum_{n=1}^N x_{n}(c_{0}-t_{n})) ...(1.1)
```

and the same with respect to $c_{0}$ yields

```math
\partial L / \partial c_{0}=2c_{0}+2c_{1} {{1}\over{N}} (\sum_{n=1}^N x_{n} )- {{2}\over{N}} (\sum_{n=1}^N t_{n}) ...(1.2)
```

The expressions need to be zero to extract the $c_{0}$ & $c_{1}$

starting from equation (**1.2**)

```math
2c_{0}+2c_{1}\frac{1}{N}(\sum_{n=1}^{N}x_{n})-\frac{2}{N}(\sum_{n=1}^{N}t_{n})=0
```
```math
2c_{0}=\frac{2}{N}(\sum_{n=1}^{N}t_{n})-c_{1}\frac{2}{N}(\sum_{n=1}^{N}x_{n})
```
```math
c_{0}=\frac{1}{N}(\sum_{n=1}^{N}t_{n})-c_{1}\frac{1}{N}(\sum_{n=1}^{N}x_{n})
```
---
The average of $t$ is 
```math
\bar{t}= 1/N \sum_{n=1}^{N}t_{n}
```
and average of $x$ is
```math
\bar{x}= 1/N\sum_{n=1}^{N}x_{n}
```
so $c_{0}$ that corresponds to the minima is expressed as

$\widehat{c_{0}}=\bar{t}-c_{1} \bar{x} ...(2)$ 


subtituting **(2)** into **(1.1)** then do some arrangement yields 

```math
\partial L / \partial c_{1} = c_{1}2/N(\sum_{n=1}^N x_{n}^2)+2/N (\sum_{n=1}^Nx_{n}(\widehat{c_{0}}-t_{n}))
```
using expression 
```math
\bar{x}=(1/N)\sum_{n=1}^{N}x_{n}
```
and gathering together $c_{1}$ terms
```math
\partial L / \partial c_{1} = 2c_{1}(1/N\sum_{n=1}^N x_{n}^2 -\bar{x}\bar{x})+2\bar{t}\bar{x} -2 (1/N)(\sum_{n=1}^Nx_{n}t_{n})
```
Setting this partial derivative to $0$ and solving $c_{1}$
```math
 2c_{1}(1/N\sum_{n=1}^N x_{n}^2 -\bar{x}\bar{x})+2\bar{t}\bar{x} -2 (1/N)(\sum_{n=1}^Nx_{n}t_{n})=0
```
```math
\widehat{c_{1}} = \frac{1/N (\sum_{n=1}^{N}x_{n}t_{n})-\bar{t}\bar{x}}
 {(\sum_{n=1}^{N}x_{n}^2)-\bar{x}\bar{x}}
 ```

Again, because the topic is about the average ($N$ data points) 

Denote  $\bar{xt}$ for the
```math
1/N (\sum_{n=1}^{N}x_{n}t_{n})
```


and $\bar{x^2}$ for the
```math
\sum_{n=1}^{N}x_{n}^2
```

therefore

```math
\widehat{c_{1}} =\frac{ \bar{xt} - \bar{x}\bar{t}}{\bar{x}^2 - (\bar{x})^2}
```

**Going Into the Matrix Formation**
---
The basic idea is the following,

Let's represent the coeffs. $c_{0}$ & $c_{1}$ as **c**

```math
c = \begin{bmatrix} c_0 \\ c_1 \end{bmatrix}
```

also with the $x$ distribution

```math
\mathbf{x}_{n}=\left[\begin{array}{c}1 \\ x_{n}\end{array}\right]
```

that $1$ makes sense in the way that to multiply them the expression required is

$c^T x_{n}$ and therefore it satisfies that **$f(x_{n};c_{0},c_{1}) = c_{0}+c_{1}x_{n}=c^T x_{n}$** 

---

Again, the loss exposure should cover the entire (N) datapoints

```math
X*c = \begin
{bmatrix}
1 & x_{1}\\
1 & x_{2}\\
.. & ..\\
1 & x_{n}\\
\end{bmatrix}
\begin
{bmatrix}
c_{0}\\
c_{1}
\end{bmatrix}
=
\begin
{bmatrix}
c_{0} +c_1x_{1}\\
c_{0} +c_1x_{2}\\
..\\
c_{0} +c_1x_{n}\\
\end{bmatrix}
```

```math
t =  \begin
{bmatrix}
t_{1}\\
t_{2}\\
..\\
t_{n}\\
\end{bmatrix}
```
and $t-Xc$ would be

```math
t = 
\begin
{bmatrix}
t_{1}- c_{0} +c_1x_{1}\\
t_{2}- c_{0} +c_1x_{2}\\
..\\
t_{n}- c_{0} +c_1x_{n}\\
\end{bmatrix}
```

and therefore the lost could be would be written as (_*notice that T (transpose) otherwise it doesn't make sense for the multiplication_)

```math
L = 1/N (t-Xc)^T * (t-Xc)
```
So overall, 

$L_{n} = t_{n}- f(x;c_{0},c_{1})^2  = 1/N (t-xc)^T * (t-xc)$ 


Now what is required is the matrix $c$ corresponding to the minima (minimum) of that $L_{n}$.This can be achieved with the partial derivative of $L_{n}$ with respect to the matrix $c$

At this point, the are still several steps involved in differentiating the $L_{n}$ but I think it's going to be another topic (or rather self explanatory if you're that math guy) as that would be too much details. 


Fortunately there are shortcuts can be applied when differentiating a vector. In short, 

$\partial{L} / \partial{c} = \frac{2}{N} x^Txc- \frac{2}{N}x^Tt = 0$ and therefore

```math
x^Txc = x^T t
```
The final step is to solve the $c$ (but it can't be done directly as no division in matrix). So the idea is using an identity matrix to eliminate that $x^Tx$ because the property of the identity tells that $Ic = c$ and therefore $Ic = (x^Tx)^{-1}x^Tt$. 

Finally, $c$ (that minimize the lost) is

$\bar{c} = (x^Tx)^{-1}x^Tt$


