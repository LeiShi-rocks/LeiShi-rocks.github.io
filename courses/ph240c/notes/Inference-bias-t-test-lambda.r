library(Rlab)
Nrep = 100000

lambda.vec = seq(-2, 0, length.out = 10)

N1 = 100
N2 = 200
theta1 = 0.5
mean.treat = matrix(data = NA, nrow = Nrep, ncol = 10)
mean.control = matrix(data = NA, nrow = Nrep, ncol = 10)

mu.treat = -1
mu.control = 0

for(j in 1:Nrep){
  for(i in 1:10){
    lambda = lambda.vec[i]
    
    d1 = rbern(N1, theta1)
    y1 = rep(0,N1)
    
    y1[d1 == 1] = rnorm(sum(d1), mu.treat, sd = 1)
    y1[d1 == 0] = rnorm(N1-sum(d1), mu.control, sd = 2)
    
    #test.stat = (mean(y1[d1 == 1]) - mean(y1[d1 == 0]))/sqrt( var(y1[d1 == 1]) + var(y1[d1 == 0]))
    #better.arm = ifelse( test.stat < lambda, 1, 0)
    
    test.stat = (mean(y1[d1 == 1]) - mean(y1[d1 == 0]))
    better.arm = ifelse( test.stat < lambda, 1, 0)
    
    theta2 = better.arm * 0.8 + (1-better.arm) * 0.2
    
    d2 = rbern(N2, theta2)
    y2 = rep(0,N2)
    
    y2[d2 == 1] = rnorm(sum(d2), mu.treat)
    y2[d2 == 0] = rnorm(N2-sum(d2), mu.control)
    
    y.treated = c(y1[d1 == 1], y2[d2 == 1])
    y.control = c(y1[d1 == 0], y2[d2 == 0])
    
    mean.treat[j, i] = mean(y.treated)  - mu.treat
    mean.control[j, i] = mean(y.control)  - mu.control
  }
}



library(tidyr)
library(ggplot2)

apply(mean.treat, 2, mean)/apply(mean.treat, 2, sd)
apply(mean.control, 2, mean)/apply(mean.control, 2, sd)

#df = data.frame(Treated = apply(mean.treat, 2, mean), 
#                Control = apply(mean.control, 2, mean))

## Standardized Bias
df = data.frame(Treated = apply(mean.treat, 2, mean)/apply(mean.treat, 2, sd), 
                Control = apply(mean.control, 2, mean)/apply(mean.control, 2, sd))

df.plot = df %>% gather(key = Group, value = Bias)
df.plot$Cutoff = c(lambda.vec, lambda.vec)


ggplot(df.plot, aes(x=Cutoff, y=Bias, color=Group)) +
  stat_smooth(aes(fill=Group, linetype=Group), method = "loess", size=0.5, se = FALSE, level = 0.95) +
  geom_point(aes(shape =Group), size = 2.5)+ 
  scale_y_continuous(limits = c(-0.1, 0.55))+
  theme_bw() +
  theme(axis.title = element_text(face="bold", size = rel(1.2)),
        axis.text = element_text( size = rel(1.2)),
        axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0, unit='pt')),
        legend.position = "bottom",
        legend.title = element_text(face = "bold", margin = margin(r = 10, unit = 'pt'), size = rel(1.2)),
        legend.text = element_text(margin = margin(r = 10, unit = 'pt'), size = rel(1.2) ) )  +
  labs(title="Standardized Bias in the Treated/Control Groups", 
       #subtitle=expression(paste("Sample drawn from ",pi,"=0.02, ", sigma[x]^2,"=", 10^{-6}  )) ,
       y = "Standardized bias magnitude", 
       x = expression(paste("Cut-off value ", lambda)))



