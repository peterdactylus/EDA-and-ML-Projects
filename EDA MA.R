library(tidyverse)
library(data.table)
library(googlesheets4)
library(lmtest)
library(sandwich)
library(ggrepel)
library(plm)

## Transformations
crt <- as.data.table(crt)
mt <- as.data.table(mt)
data <- merge(mt, crt, by = "responseId")
data <- data[totalTasksAttempted > 0]
data[, bailedTask := fifelse(totalTimeElapsed == "NaN", 1, 0)]
data[, perfectScoreGrid := 
       fifelse(totalTasksAttempted == totalTasksAnsweredCorrectly, 1, 0)]
data[, accuracy_score := totalTasksAnsweredCorrectly / totalTasksAttempted]
data[, accuracy_perc := percent_rank(accuracy_score)]
data[, precision := accuracy_score * totalTasksAnsweredCorrectly]
data[, above_average_accuracy := fifelse(accuracy_perc > 0.5, 
                                         "Above Average", "Below Average")]
data[, average_task_time := totalTaskTimeElapsed / totalTasksAttempted]
data[, average_task_time_perc := percent_rank(average_task_time)]
data[, crt_result := batBallSuccess + machineWidgetSuccess + lilyPadSuccess]

x <- seq(19, 373, 6)
x <- append(x, 1)
main_task <- data[, ..x]
rm(x)
main_task <- melt(main_task, id.vars = "responseId")
main_task[, variable := gsub("taskTimeElapsed", "", variable)]
main_task <- main_task[!is.na(value) & value != "NULL"]
main_task$variable <- as.numeric(main_task$variable)
main_task[, percentile := percent_rank(value)]

x <- seq(381, 499, 2)
x <- append(x, 1)
tmp <- data[, ..x]
rm(x)
tmp <- melt(tmp, id.vars = "responseId")
tmp <- na.omit(tmp)
tmp <- tmp[, sum(value), by = "responseId"]
data <- merge(data, tmp, by = "responseId", all.x = T)
setnames(data, "V1", "attention_score")
data[, attention_score := attention_score / 60]

x <- seq(20, 374, 6)
x <- append(x, 1)
tmp <- data[, ..x]
rm(x)
tmp <- melt(tmp, id.vars = "responseId")
tmp[, variable := gsub("taskSuccess", "", variable)]
tmp <- na.omit(tmp)
tmp$variable <- as.numeric(tmp$variable)
main_task <- merge(main_task, tmp, by = c("responseId", "variable"))

main_task <- merge(main_task, 
                   data[, c("responseId", "attention_score", "crt_result", 
                            "bailedTask")], by = "responseId", all.x = T)
main_task[, attention_perc := percent_rank(attention_score)]
main_task[, attention_class := fifelse(attention_perc < 0.5, 0, 1)]
main_task[, crt_class := fifelse(crt_result > 1, 2, crt_result)]

shapiro.test(data[, precision])

      #Average Task Time based on bailed or not
wilcox.test(data[bailedTask == 0 & average_task_time_perc > 0.025 &
                   average_task_time_perc < 0.975, average_task_time], 
            data[bailedTask == 1 & average_task_time_perc > 0.025 &
                   average_task_time_perc < 0.975, average_task_time])

      #Accuracy and bailed Task Ratio comparison
ct <- table(data[, c("bailedTask", "above_average_accuracy")])
ct
fisher.test(ct)

      #Task Time First Task and Bail Rate
ggplot(data, aes(as.factor(bailedTask), taskTimeElapsed1)) +
  geom_boxplot() +
  scale_y_log10() +
  theme_bw()

wilcox.test(data[bailedTask == 0, taskTimeElapsed1], 
            data[bailedTask == 1, taskTimeElapsed1])

## Task Time Analysis
      #Summary Statistics of Time per individual Task
summary(main_task[percentile > 0.025 & percentile < 0.975, value.x])

ggplot(main_task[percentile > 0.025 & percentile < 0.975], 
       aes(variable, value.x)) + 
  geom_point(alpha = 0.25) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2)) +
  theme_bw() + 
  labs(title = "Task Time Decreases With More Task Iterations", 
       x = "Iteration", y = "Task Time") +
  theme(text = element_text(family = "serif", size = 14))


    #30_09
course <- function(person) {
  tmp <- main_task[responseId == person]
  result <- c()
  
  for(i in 1:nrow(tmp)) {
    result <- append(result, sum(tmp[c(1:i)]$value.x))
  }
  return(data.table(person, tmp[, variable], result))
}

result <- lapply(main_task[bailedTask != 1, unique(responseId)], 
                 function(x){course(x)})
trend <- rbindlist(result)
rm(result)
trend[, minute := fifelse(result < 60, 1, 
                          fifelse(result < 120, 2, 
                                  fifelse(result < 180, 3, 
                                          fifelse(result < 240, 4, 5))))]
#trend[, result := fifelse(result < 60, result, 
                          #fifelse(result < 120, result - 60, 
                                  #fifelse(result < 180, result - 120, 
                                          #fifelse(result < 240, result - 180, 
                                                  #result - 240))))]

trend[, .N, by = c("minute", "person")] %>%
  ggplot(aes(as.factor(minute), N)) +
  geom_boxplot() +
  theme_bw() +
  labs(y = "Iterations", x = "Minute")

tmp <- trend[, .N, by = c("minute", "person")]
wilcox.test(tmp[minute == 1, N], tmp[minute == 2, N])

      #Task Time predicted by Iteration (linear and quadratic)
summary(lm(value.x ~ variable, data = main_task))

summary(plm(value.x ~ variable, model = "within", index = "responseId",
            data = main_task))

ols <- lm(value.x ~ poly(variable, 2), 
          data = main_task[percentile > 0.025 & percentile < 0.975])
summary(ols)

fixed <- plm(value.x ~ poly(variable, 2), model = "within", 
             index = "responseId", 
             data = main_task[percentile > 0.025 & percentile < 0.975])

fixed <- lm(value.x ~ poly(variable, 2) + factor(responseId) - 1,  
             data = main_task[percentile > 0.025 & percentile < 0.975])
summary(fixed)
pFtest(fixed, ols)

fixed_pred <- main_task[percentile > 0.025 & percentile < 0.975, 
                        c("responseId", "variable")]
fixed_pred[, pred_fix := predict(fixed)]
fixed_pred[, pred_ols := predict(ols)]

fixed_pred[, sum(pred_ols) / .N, by = "variable"] %>%
ggplot(aes(variable, V1)) +
  geom_point() +
  theme_bw() +
  labs(y = "predicted avg dwell time", x = "iteration")

      #Test for Heteroscedasticity
test <- lm(variable ~ value.x, 
           data = main_task[percentile > 0.025 & percentile < 0.975])
bptest(test)
test2 <- lm(variable ~ poly(value.x, 2), 
            data = main_task[percentile > 0.025 & percentile < 0.975])
bptest(fixed)

      #Test Coefficients with Heteroscedasticity consistent standard errors
coeftest(test, vcov. = vcovHC(test))
coeftest(fixed, vcov. = vcovHC(fixed))
rm(test, test2)

      #Accuracy predicted by Average Task Time (linear)
summary(lm(accuracy_score ~ average_task_time, data = data))

      #Precision predicted by Average Task Time (linear and quadratic)
summary(lm(precision ~ average_task_time, data = data))
summary(lm(precision ~ poly(average_task_time, 2), data = data))

ggplot(data[average_task_time > 0.025 & average_task_time_perc < 0.975], 
       aes(average_task_time, precision)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2)) + 
  theme_bw() + 
  labs(title = "Performance Decreases with higher Average Task Times", 
       x = "Average Task Time", y = "Performance") +
  theme(text = element_text(family = "serif", size = 14))

      #Average Task Time based on above or below average accuracy
shapiro.test(data[average_task_time_perc > 0.025 & 
                    average_task_time_perc < 0.975, average_task_time])
wilcox.test(data[average_task_time_perc > 0.025 & 
                   average_task_time_perc < 0.975 & 
                   above_average_accuracy == "Below Average", average_task_time], 
            data[average_task_time_perc > 0.025 & 
                   average_task_time_perc < 0.975 & 
                   above_average_accuracy == "Above Average", average_task_time])

## Task Accuracy over Iterations
      #Task Time based on correct or incorrect answer
wilcox.test(main_task[percentile > 0.025 & percentile < 0.975 & 
                        value.y == 0, value.x], 
            main_task[percentile > 0.025 & percentile < 0.975 & 
                        value.y == 1, value.x])

ggplot(main_task[percentile > 0.025 & percentile < 0.975], 
       aes(as.factor(value.y), value.x)) +
  geom_boxplot() +
  theme_bw()

main_task[variable < 41, sum(value.y) / .N, by = "variable"] %>%
  ggplot(aes(variable, V1)) +
  geom_point() + 
  geom_smooth(method = "lm", formula = y ~ poly(x, 2)) +
  theme_bw() + 
  labs(title = "Overall Accuracy per Task Iteration", 
       subtitle = "For Iterations where n >= 20", x = "Iteration", 
       y = "Accuracy") +
  theme(text = element_text(family = "serif", size = 14))

main_task[variable < 41 & 
            percentile > 0.025 & percentile < 0.975, 
          .(accuracy = sum(value.y) / .N, dwell_time = sum(value.x) / .N), 
          by = "variable"] %>%
  ggplot(aes(dwell_time, accuracy, label = variable)) +
  geom_point() + 
  geom_text_repel() +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2)) +
  theme_bw() + 
  labs(title = "Overall Accuracy per Task Iteration", 
       subtitle = "For Iterations where n >= 20", x = "Dwell Time", 
       y = "Accuracy") +
  theme(text = element_text(family = "serif", size = 14))

main_task[, .(accuracy = sum(value.y) / .N, sample = .N), 
          by = c("variable", "crt_class")] %>% .[sample > 4] %>%
  ggplot(aes(variable, accuracy)) +
  geom_point() + 
  geom_smooth(method = "lm", formula = y ~ poly(x, 2)) +
  theme_bw() +
  facet_wrap(~crt_class) +
  labs(title = "Overall Accuracy per Task Iteration", 
       subtitle = "For Iterations where n >= 5", x = "Iteration", 
       y = "Accuracy") +
  theme(text = element_text(family = "serif", size = 14))

summary(lm(accuracy ~ variable + I(variable^2), 
           data = main_task[, .(accuracy = sum(value.y) / .N, sample = .N), 
                            by = c("variable", "crt_class")][sample > 4 &
                                                               crt_class == 2]))

main_task[, .(accuracy = sum(value.y) / .N, sample = .N), 
          by = c("variable", "attention_class")] %>% 
  .[sample > 4 & !is.na(attention_class)] %>%
  ggplot(aes(variable, accuracy)) +
  geom_point() + 
  geom_smooth(method = "lm", formula = y ~ poly(x, 2)) +
  theme_bw() +
  facet_wrap(~attention_class) +
  labs(title = "Overall Accuracy per Task Iteration", 
       subtitle = "For Iterations where n >= 5", x = "Iteration", 
       y = "Accuracy") +
  theme(text = element_text(family = "serif", size = 14))

      #Accuracy predicted by Iteration (quadratic)    
summary(lm(V1 ~ poly(variable, 2), 
           data = main_task[variable < 41, sum(value.y) / .N, by = "variable"]))
summary(lm(accuracy ~ poly(variable, 2) + factor(responseId) - 1, 
            data = main_task[variable < 41, 
                             .(accuracy = sum(value.y) / .N,
                               responseId = responseId), by = "variable"]))

## CRT Performance
ggplot(data, aes(as.factor(crt_result), precision)) +
  geom_boxplot() +
  theme_bw() + 
  labs(title = "CRT Score >0 Indicates Better Performance", 
       x = "CRT Questions Answered Correctly", y = "Performance") +
  theme(text = element_text(family = "serif", size = 14))

      #Performance based on CRT questions answered correctly
t.test(data[crt_result == 0, precision], data[crt_result == 1, precision])

ggplot(data, aes(as.factor(crt_result), fill = as.factor(above_average_accuracy))) +
  geom_bar(position = "fill") +
  theme_bw() +
  labs(fill = "Accuracy", 
       x = "CRT Questions Answered Correctly", y = "Ratio", 
       title = "Perfect CRT Result Indicates Above Average Accuracy") +
  theme(text = element_text(family = "serif", size = 14))

      #Testing for significant difference in distribution of accuracy and two CRT results
ct <- table(data[crt_result == 0 | crt_result == 3, 
                 c("crt_result", "above_average_accuracy")])
ct
fisher.test(ct)

## Attention Performance
shapiro.test(data[, attention_score])

ggplot(data, aes(attention_score, precision)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm") + 
  theme_bw() + 
  labs(title = "Higher Attention Score predicts better Performance", 
       x = "Attention Score", y = "Performance") +
  theme(text = element_text(family = "serif", size = 14))

      #Performance predicted by Attention Score
summary(lm(precision ~ attention_score, data = data))

ggplot(data, aes(as.factor(above_average_accuracy), attention_score)) +
  geom_boxplot() +
  theme_bw() + 
  labs(title = "Above Average Accuracy Indicates Higher Attention Score", 
       x = "Accuracy", y = "Attention Score") +
  theme(text = element_text(family = "serif", size = 14))

wilcox.test(data[above_average_accuracy == "Above Average", attention_score], 
            data[above_average_accuracy == "Below Average", attention_score])

## Subjective Performance
data[, subjectivePerformanceAccuracy := 
       gsub("Strongly agree", "5", subjectivePerformanceAccuracy)]
data[, subjectivePerformanceAccuracy := 
       gsub("Agree", "4", subjectivePerformanceAccuracy)]
data[, subjectivePerformanceAccuracy := 
       gsub("Neither agree nor disagree", "3", subjectivePerformanceAccuracy)]
data[, subjectivePerformanceAccuracy := 
       gsub("Disagree", "2", subjectivePerformanceAccuracy)]
data[, subjectivePerformanceAccuracy := 
       gsub("Strongly disagree", "1", subjectivePerformanceAccuracy)]
data$subjectivePerformanceAccuracy <- as.numeric(data$subjectivePerformanceAccuracy)

data[, subjectivePerformanceSpeed := 
       gsub("Strongly agree", "5", subjectivePerformanceSpeed)]
data[, subjectivePerformanceSpeed := 
       gsub("Agree", "4", subjectivePerformanceSpeed)]
data[, subjectivePerformanceSpeed := 
       gsub("Neither agree nor disagree", "3", subjectivePerformanceSpeed)]
data[, subjectivePerformanceSpeed := 
       gsub("Disagree", "2", subjectivePerformanceSpeed)]
data[, subjectivePerformanceSpeed := 
       gsub("Strongly disagree", "1", subjectivePerformanceSpeed)]
data$subjectivePerformanceSpeed <- as.numeric(data$subjectivePerformanceSpeed)

      #Average Task Time predicted by Subjective Performance Speed
summary(lm(average_task_time ~ subjectivePerformanceSpeed, 
           data[average_task_time_perc > 0.025 & average_task_time_perc < 0.975]))

      #Subjective Performance Accuracy based on Above and Below Average Score
wilcox.test(data[above_average_accuracy == "Above Average", subjectivePerformanceAccuracy], 
            data[above_average_accuracy == "Below Average", subjectivePerformanceAccuracy])

## Pre Task
pre <- as.data.table(pre)
x <- seq(19, 373, 6)
y <- seq(20, 374, 6)
x <- append(x, c(1, 18))
y <- append(y, c(1, 18))
pre_time <- pre[, ..x]
pre_acc <- pre[, ..y]
rm(x, y)
pre_time <- melt(pre_time, id.vars = c("responseId", "taskMetaData1"))
pre_acc <- melt(pre_acc, id.vars = c("responseId", "taskMetaData1"))
pre_time[, variable := gsub("taskTimeElapsed", "", variable)]
pre_acc[, variable := gsub("taskSuccess", "", variable)]
pre_acc <- na.omit(pre_acc)
pre_time <- na.omit(pre_time)
pre_acc$variable <- as.numeric(pre_acc$variable)
pre_time$variable <- as.numeric(pre_time$variable)
setnames(pre_acc, "value", "accuracy")
setnames(pre_time, "value", "time")
pre_task <- cbind(pre_acc, pre_time[, "time"])
rm(pre_acc, pre_time)
pre_task[, time_perc := percent_rank(time), by = "taskMetaData1"]

pre_task[, taskMetaData1 := gsub("4", "4 | EASY", taskMetaData1)]
pre_task[, taskMetaData1 := gsub("5", "5 | MEDIUM", taskMetaData1)]
pre_task[, taskMetaData1 := gsub("6", "6 | HARD", taskMetaData1)]

pre_task[time_perc > 0.025 & time_perc < 0.975, sample := .N, 
         by = c("taskMetaData1", "variable")]

      #Sample Size (Participants per Difficulty)
pre_task[taskMetaData1 == "4 | EASY", uniqueN(responseId)]
pre_task[taskMetaData1 == "5 | MEDIUM", uniqueN(responseId)]
pre_task[taskMetaData1 == "6 | HARD", uniqueN(responseId)]

      #Participants with at least one valid task iteration
pre_task[taskMetaData1 == "4 | EASY" & 
           time_perc > 0.025 & time_perc < 0.975, uniqueN(responseId)]
pre_task[taskMetaData1 == "5 | MEDIUM" & 
           time_perc > 0.025 & time_perc < 0.975, uniqueN(responseId)]
pre_task[taskMetaData1 == "6 | HARD" & 
           time_perc > 0.025 & time_perc < 0.975, uniqueN(responseId)]

      #Summary Statistics Completion Time
summary(pre_task[taskMetaData1 == "4 | EASY" & 
                   time_perc > 0.025 & time_perc < 0.975, time])
summary(pre_task[taskMetaData1 == "5 | MEDIUM" & 
                   time_perc > 0.025 & time_perc < 0.975, time])
summary(pre_task[taskMetaData1 == "6 | HARD" & 
                   time_perc > 0.025 & time_perc < 0.975, time])

      #Summary Statistics Accuracy      
summary(pre_task[taskMetaData1 == "4 | EASY" & 
                   time_perc > 0.025 & time_perc < 0.975, accuracy])
summary(pre_task[taskMetaData1 == "5 | MEDIUM" & 
                   time_perc > 0.025 & time_perc < 0.975, accuracy])
summary(pre_task[taskMetaData1 == "6 | HARD" & 
                   time_perc > 0.025 & time_perc < 0.975, accuracy])

      #Mean Grids attempted per Participant
pre_task[, .N, by = c("responseId", "taskMetaData1")] %>%
.[, sum(N) / .N, by = "taskMetaData1"]

ggplot(pre_task[time_perc > 0.025 & time_perc < 0.975 & sample > 4], 
       aes(variable, time)) +
  geom_point(alpha = 0.25) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2)) +
  theme_bw() +
  facet_wrap(~taskMetaData1, scales = "free_x") +
  labs(title = "Task Time per Iteration for Different Complexity Levels", 
       subtitle = "For Iterations where n >= 5", 
       x = "Iteration", y = "Task Time") +
  theme(text = element_text(family = "serif", size = 14), 
        strip.background = element_rect(fill = "lightblue1"))

pre_task[time_perc > 0.025 & time_perc < 0.975 & sample > 4, 
         sum(accuracy) / .N, by = c("variable", "taskMetaData1")] %>%
ggplot(aes(variable, V1)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2)) +
  theme_bw() +
  facet_wrap(~taskMetaData1, scales = "free_x") +
  labs(title = "Overall Accuracy per Iteration for Different Complexity Levels",
       subtitle = "For Iterations where n >= 5",
       x = "Iteration", y = "Accuracy") +
  theme(text = element_text(family = "serif", size = 14), 
        strip.background = element_rect(fill = "lightblue1"))

ggplot(data, aes(attention_score, totalTasksAttempted)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_bw()

summary(lm(totalTasksAttempted ~ attention_score, data = data))

ggplot(data, aes(as.factor(crt_result), attention_score)) +
  geom_boxplot() +
  theme_bw()

shapiro.test(data[, attention_score])

wilcox.test(data[crt_result == 2, attention_score], 
            data[crt_result == 3, attention_score])

ggplot(data[average_task_time_perc > 0.025 & 
              average_task_time_perc < 0.975 & accuracy_inv < 6], 
       aes(accuracy_inv)) +
  geom_density() +
  theme_bw()


summary(lm(accuracy ~ dwell_time + variable, 
           data = main_task[variable < 41 & variable > 1 &
                              percentile > 0.025 & percentile < 0.975, 
                              .(accuracy = sum(value.y) / .N, 
                                dwell_time = sum(value.x) / .N), 
                                                   by = "variable"]))

ggplot(main_task[percentile > 0.025 & percentile < 0.975], 
       aes(as.factor(value.y), value.x)) +
  geom_boxplot() +
  theme_bw() +
  labs(x = "-----", y = "Dwell Time", 
       title = "------") +
  theme(text = element_text(family = "serif", size = 14))

wilcox.test(main_task[percentile > 0.025 & percentile < 0.975 & 
                        value.y == 1, value.x], 
            main_task[percentile > 0.025 & percentile < 0.975 & 
                        value.y == 0, value.x])

data[, accuracy_inv := 11 - accuracy_score*10]

summary(glm(accuracy_inv ~ average_task_time, 
            data = data[average_task_time_perc > 0.025 & 
                          average_task_time_perc < 0.975], 
            family = Gamma))


ggplot(data, aes(as.factor(crt_result), attention_score)) +
  geom_boxplot() +
  theme_bw()

wilcox.test(data[crt_result == 0, attention_score], 
            data[crt_result == 3, attention_score])
