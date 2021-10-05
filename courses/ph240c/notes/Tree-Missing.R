if (!require("rpart")){
  install.package("rpart")
}

tmp_df <- 
  data.frame(Y = as.factor(c(1, 1, 1, 1, 1, 0, 0, 0, 0, 0)),
             weight = 10:1,
             height = c(10:7, 5, 6, 4:1))

tmp_df$weight[3] <- NA
tmp_df


tm_0 <- rpart(Y ~ weight + height, data = tmp_df, 
              control = rpart.control(minsplit = 1,
                                      minbucket=1, 
                                      cp=0, 
                                      maxdepth = 1, 
                                      usesurrogate = 0))
summary(tm_0)


tm <- rpart(Y ~ weight + height, data = tmp_df, 
            control = rpart.control(minsplit =1,
                                    minbucket=1, 
                                    cp=0, 
                                    maxdepth = 1))
summary(tm)

predict(tm_0, newdata = tmp_df)
predict(tm, newdata = tmp_df)


tmp_new_df <- data.frame(weight = c(rep(NA_real_, 4), 3:6), height = rep(3:6, 2))
tmp_new_df
predict(tm_0, newdata = tmp_new_df)
predict(tm, newdata = tmp_new_df)


