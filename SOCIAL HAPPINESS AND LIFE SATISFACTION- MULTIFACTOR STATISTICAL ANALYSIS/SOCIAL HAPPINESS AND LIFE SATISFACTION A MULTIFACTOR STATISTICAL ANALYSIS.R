########## TASK 1 ##########

# We fit a 5 factor model with correlated latent variables 
# "Social Trust, Satisfaction COuntry, Job Autonomy, Job Satisfaction, Life Satisfaction"


CFA1<-'SocialTrust =~1*pp_trusted+pp_fair+
pp_helpful

 SatisfactionC=~1*satisf_economy+satisf_government+satisf_democracy

 JobAutonomy=~1*decide_work+influence_deciswork
 
 JobSatisfaction=~1*satisf_job+satisf_worklife
 
 LifeSatisfaction=~1*satisflife+howhappy

 SocialTrust~~SocialTrust
 SatisfactionC~~SatisfactionC
 JobAutonomy~~JobAutonomy
 JobSatisfaction~~JobSatisfaction
 LifeSatisfaction~~LifeSatisfaction
 
 SatisfactionC~~SocialTrust+JobAutonomy+JobSatisfaction+LifeSatisfaction
 LifeSatisfaction~~JobSatisfaction
 JobAutonomy~~JobSatisfaction
 SatisfactionC~~JobSatisfaction+LifeSatisfaction
 '

library(lavaan)

#fit model on covariance matrix
fitCFA1<-cfa(CFA1,ess2)
#summary of results
summary(fitCFA1,fit.measures=TRUE)

#print fitmeasures
fitmJ1<-fitmeasures(fitCFA1,
                    c("chisq","df","pvalue","cfi","tli","rmsea","srmr"))
fitmJ1 


## Our model is rejected by an absolute goodness of fit test (Chi_Squared=174.9,
# df= 44, p<0.000) However as the sample size is large this iis not problematic since the high statistical 
# power to detect a small deviation between the fitted model
## and the perfect model. Therefore it is appropriate to rely on  descriptive measures
# The printed fit measures indicate that all measures meet the cutoff for good fit with CFI (0.985) and 
# TLI (0.977) both greater than 0.95 and RMSEA (0.04) and SRMR (0.026) both less that 0.08.

## We now look at the standardized solutions#

stan<-standardizedSolution(fitCFA1)
head(stan)

#As shown in the standardized solution all variables have a significant positive correlation of
#at least .70 with the corresponding factor. This means that the individual variables have
#sufficient reliability, and that convergent validity is satisfied in the measurement model.
#Furthermore, discriminant validity is also satisfied as the correlations between the latent
#factors are all significantly smaller than 1 . Note that there is a rather strong correlation
#(.635) between the factors "Social Trust" and "Satisfaction Country". This makes sense
#because, a higher score on both factors means that one who can trust there society can be satisfies with their 
#life in a country. All other factors have weaker but positive correlations between each other.

# From the output of the standardized solution we can see that PP(trusted, fair, helpful) have reliablities 
# respectively 0.603 0.628 and 0.49. The Country satisfaction variables have ok reliabilities respectively 0.653, 0.755 
# and 0.63. The Job autonomy variables have reliabilities 0.812 and 0.551. The Job satisfaction variables have reliabilities 
# 0.66 and 0.308 and the life satisfaction variables have reliabilities 0.861 and 0.607.

# Now to fit multi-group models to investigate measurement invariance.

config<-cfa(CFA1,data=ess2,group="country")
metric<-cfa(CFA1,data=ess2,group="country",group.equal="loadings")
strong<-cfa(CFA1,data=ess2,group="country",
            group.equal=c("loadings","intercepts")) 

#summarize fitmeasures
fitconfig<-
  fitmeasures(config,c("chisq","df","pvalue","cfi","tli","rmsea","srmr")) 

fitmetric<-fitmeasures(metric,
                       c("chisq","df","pvalue","cfi","tli","rmsea","srmr"))
fitstrong<-fitmeasures(strong,
                       c("chisq","df","pvalue","cfi","tli","rmsea","srmr"))
fit1<-rbind(fitconfig,fitmetric,fitstrong)
rownames(fit1)<-c("configural","metric","strong")
round(fit1,3) 

#compare models using LR test
anova(config,metric) 
anova(config,strong) 

#The results of the analysis indicate that the fit of the configural model is similar to
#the fit obtained for the total data set. The configural model does meet the cutoff criteria
#of good fit, i.e. CFI and TLI are much above .95, and RMSE and MRSR are below .08.

#We also see that imposing further model constraints (assume equal loadings across groups,
#or assume equal loadings and intercepts across groups) further reduces the fit of the model but still
# meets the cut-off criteria.

#A LR test shows that metric measurement invariance is not supported by the data
#(LR=49.724, df=7, p<.001). In the same way, strong measurement invariance is not
#supported by the data (LR=112.84, df=14, p<.001).

#As the results of the analysis indicate that metric or strong measurement equivalence do not
#hold, we print the standardized loadings, the intercepts and the factor correlations for the
#two countries from the standardized solution of the configural model.

stan<-standardizedSolution(config)
stan
#loadings
load<-cbind(stan[1:12,c(1:3,5)],stan[57:68,c(5)])
colnames(load)[4:5]<-c("Poland","Sweden")
print(load,digits=3) 

#correlations
#Poland
print(stan[18:23,c(1:3,5:8)],digits=3) 

#Sweden
print(stan[74:79,c(1:3,5:8)],digits=3) 

#intercepts
int<-cbind(stan[40:51,c(1:3,5)],stan[96:107,c(5)])
colnames(int)[4:5]<-c("Poland","Sweden")
print(int,digits=3) 

# Both Countries have similar standardized loading 
# implying that both countries have similar composite reliability.

# All factor correlations are overall weak but positive in both Poland 
# and Sweden. 

# This I Imagine means we have too many factors involved in this model.
# To fix this problem I would suggest grouping Satisfaction of country and life together
# and grouping job Autonomy and Job satisfaction together.



semming1 <- '
 # latent variable definitions
 SocialTrust =~1*pp_trusted+pp_fair+pp_helpful
 SatisfactionC=~1*satisf_economy+satisf_government+satisf_democracy
 JobAutonomy=~1*decide_work+influence_deciswork
 JobSatisfaction=~1*satisf_job+satisf_worklife
 LifeSatisfaction=~1*satisflife+howhappy

 JobSatisfaction~JobAutonomy

# regressions
 SocialTrust~LifeSatisfaction
 SatisfactionC~LifeSatisfaction
 JobAutonomy~LifeSatisfaction 
 JobSatisfaction~LifeSatisfaction
 '

fittedsem1 <- sem(semming1, data = ess2, group = 'country')
fitmeasures(fittedsem1,c("chisq","df","pvalue","cfi","tli","rmsea","srmr")) 

# As before all our fit measures make the cut off for good fit and model is rejected
# by an absolute goodness of fit test as expected.

# Next we use the standardized loadings to compute the composite reliability of the factor
# scores for each country:


factor<-c("SocialTrust","SatisfactionCountry","JobAutonomy","JobSatisfaction","LifeSatisfaction")
crPoland<-c(
  compositereliability(stan[1:3,5]),
  compositereliability(stan[4:6,5]),
  compositereliability(stan[7:8,5]),
  compositereliability(stan[9:10,5]),
  compositereliability(stan[11:12,5]))
crSwenden<-c(
  compositereliability(stan[57:59,5]),
  compositereliability(stan[60:62,5]),
  compositereliability(stan[63:64,5]),
  compositereliability(stan[65:66,5]),
  compositereliability(stan[67:68,5]))
data.frame(factor,Poland=round(crPoland,3),
           Sweden=round(crSwenden,3)) 

## Here we see Poland has some excellent composite reliabilities in SatisfactionCountry, 
# JobAutonomy and LifeSatisfaction whereas the composite reliabilities for Job Satisfaction and Social Trust 
# are just ok. 

# Sweden has 3 good composite reliabilities in Social Trust, Satisfaction Country and Job Autonomy, 1 
# excellent Cr in Life Satisfaction and Job Satisfaction has an ok CR.

#Next, we inspect the standardized solution to see how the latent variables affect the life satisfaction of the conutry
#factors in both countries: 

stansem<-standardizedsolution(fittedsem1)

#regressions Malaysia
stansem[14:17,1:8]

#regressions The Netherlands
stansem[68:71,1:8] 

# The results of the structural relations indicate that in Poland, 
# Life Satisfaction has a weak but significant positive correlations with
# each of the other 4 factors. Hence people who are satisfied in life have greater 
# trust in society, job autonomy and are satisfied by their job and country.

## Sweden also shows a weak but significant positive correlations between Life Satisfaction
# and the rest of the factors.

# Now we conduct multi-group analysis to investigate whether the results of the structural
#equation model differ for the two countries.

fittedsem2 <- sem(semming1, data =ess2,
                  group="country",group.equal="regressions")
fitmeasures(fittedsem2,
            c("chisq","df","pvalue","cfi","tli","rmsea","srmr"))
anova(fittedsem1,fittedsem2) 

# As could be expected, a LR test indicates that assuming equal regression coefficients across
# the two countries is not supported by the data


config<-sem(semming1,data=ess2,group="country")
metric<-sem(semming1,data=ess2,group="country",group.equal="loadings")
strong<-sem(semming1,data=ess2,group="country",
            group.equal=c("loadings","intercepts"))

fitconfig<-
  fitmeasures(config,c("chisq","df","pvalue","cfi","tli","rmsea","srmr")) 

fitmetric<-fitmeasures(metric,
                       c("chisq","df","pvalue","cfi","tli","rmsea","srmr"))
fitstrong<-fitmeasures(strong,
                       c("chisq","df","pvalue","cfi","tli","rmsea","srmr"))
fit3<-rbind(fitconfig,fitmetric,fitstrong)
rownames(fit3)<-c("configural","metric","strong")
round(fit3,3) 

#compare models using LR test
anova(config,metric) 
anova(config,strong) 

# Multi-group Analysis
#The results of the analysis indicate that the fit of the configural model is similar to
#the fit obtained for the total data set. The configural model meets the cutoff criteria
#of good fit, i.e. CFI and TLI are much above .95, and RMSE and MRSR are below .08.

#We also see that imposing further model constraints (assume equal loadings across groups,
#or assume equal loadings and intercepts across groups) further reduces the fit of the model but still
# meets the cut-off criteria.

#A LR test shows that metric measurement invariance is not supported by the data
#(LR=46.413, df=7, p<.001). In the same way, strong measurement invariance is not
#supported by the data (LR=109.62, df=14, p<.001).

#As the results of the analysis indicate that metric or strong measurement equivalence do not
#hold, we print the standardized loadings, the intercepts and the factor correlations for the
#two countries from the standardized solution of the configural model.


########## TASK 2 ##########

setwd("/Users/ruifangge/Desktop/MS/assignment")
load("dtrust.Rdata")

#load library
library(candisc)
#standardize variables
zdtrust<-dtrust
zdtrust[,2:11]<-scale(dtrust[,2:11],center=TRUE,scale=TRUE)

#conduct canonical correlation analysis
cancor.out<-cancor(cbind(satisf_economycntry,
                         satisf_nationalgovernment,
                         satisf_democracycntry)~trust_police+
                     trust_politicians+
                     trust_cntryparliament+
                     trust_legalsystem+
                     trust_politicalparties+
                     trust_EUparliament+
                     trust_UN,
                   data=zdtrust)
summary(cancor.out)

#compute redundancies
redu<-redundancy(cancor.out)
round(redu$Ycan.redun,6)
R2tu<-cancor.out$cancor^2
VAFYbyt<-apply(cancor.out$structure$Y.yscores^2,2,sum)/3
redund<-R2tu*VAFYbyt
round(cbind(R2tu,VAFYbyt,redund,total=cumsum(redund)),4)



#canonical loadings
round(cancor.out$structure$X.xscores,2)
round(cancor.out$structure$Y.yscores,2)

plot(-5,-5,xlim=c(-2.2,2.2),ylim=c(-2.2,2.2),xlab="u1",ylab="t1")
points(cancor.out$scores$X[zdtrust$country=="Poland",1],cancor.out$scores$Y[zdtrust$country=="Poland",1],col="red")
points(cancor.out$scores$X[zdtrust$country=="Sweden",1],cancor.out$scores$Y[zdtrust$country=="Sweden",1],col="blue")
legend("topleft",c("Poland","Sweden"),col=c("red","blue"),pch=c(1,1))

################################################
#split data in two parts and standardize data
################################################
train<-dtrust[seq(2,3040,by=2),]
valid<-dtrust[seq(1,3040,by=2),]
train[,2:11]<-scale(train[,2:11],center=TRUE,scale=TRUE)
valid[,2:11]<-scale(valid[,2:11],center=TRUE,scale=TRUE)

#conduct CCA on calibration data
cancor.train<-cancor(cbind(satisf_economycntry,
                           satisf_nationalgovernment,
                           satisf_democracycntry)~trust_police+
                       trust_politicians+
                       trust_cntryparliament+
                       trust_legalsystem+
                       trust_politicalparties+
                       trust_EUparliament+
                       trust_UN,
                     data=train)
summary(cancor.train)
cancor.train$structure$X.xscores
cancor.train$structure$Y.yscores

#conduct CCA on validation data
cancor.valid<-cancor(cbind(satisf_economycntry,
                           satisf_nationalgovernment,
                           satisf_democracycntry)~trust_police+
                       trust_politicians+
                       trust_cntryparliament+
                       trust_legalsystem+
                       trust_politicalparties+
                       trust_EUparliament+
                       trust_UN,
                     data=valid)
summary(cancor.valid)
cancor.valid$structure$X.xscores
cancor.valid$structure$Y.yscores

# canonical variates calibration set
train.X1<-cancor.train$score$X
train.Y1<-cancor.train$score$Y

# compute canonical variates using data of calibration set and coefficients estimated on validation set
train.X2<-as.matrix(train[,2:8])%*%cancor.valid$coef$X
train.Y2<-as.matrix(train[,9:11])%*%cancor.valid$coef$Y


#R(T,T*) and R(U,U*)
round(cor(train.Y1,train.Y2)[1:3,1:3],3)
round(cor(train.X1,train.X2)[1:3,1:3],3)

#R(U*,T*) versus R(U,T)
round(cor(train.X1,train.Y1)[1:3,1:3],3)
round(cor(train.X2,train.Y2)[1:3,1:3],3)

#R(T*,T*) and R(U*,U*)
round(cor(train.Y2,train.Y2)[1:3,1:3],3)
round(cor(train.X2,train.X2)[1:3,1:3],3)

round(cancor.out$structure$X.xscores[,1],2)
round(cancor.out$structure$Y.yscores[,1],2)
#R(T,T*) and R(U,U*)
round(cor(train.Y1,train.Y2)[1:3,1:3],3)
round(cor(train.X1,train.X2)[1:3,1:3],3)

