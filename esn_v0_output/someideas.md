## Turn NDL into CFGP (aka context-free grammar parser)?

::: warning
*Quote: "These findings imply that it is impossible to train ESNs on tasks which require unbounded-time memory, like for instance context-free grammar parsing tasks (Schmidhuber et al. 2007)." (Jaeger, 2007, Scholarpaedia)*
:::

# Ideas:
+ if the actual training list is ABCDE 
+ the training input is ABC (a many dims as possibly can, but always include "contextless" changes of notes from n to n+1)
+ then the training output may be hot_code{ABCD} or hot_code{notechange{ABCD}}, which approximates a symbolic grammar parser, based on the idea of NDL

# other things to consider
+ how should we regularize the spectrum radius
  - or how can we regularize how long the broadcast input in a specific node would fade out
  - should we chose some not-fading-out-radius like +- .99999, or always fade out neurons like +- .00001
+ how should we optimize the pars. (e.g., [some_exp](https://link-springer-com.proxy-ub.rug.nl/article/10.1007/s10489-019-01546-w))
