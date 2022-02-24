library(tidyverse)
library(data.table)
library(nflfastR)
library(ggrepel)

future::plan("multisession")
data <- load_pbp(2017:2021)
data <- as.data.table(data)

#Passing EPA Adjustment
adj <- lm(epa ~ down + yardline_100 + ydstogo + yards_gained + first_down, 
          data = data[interception == 0 & fumble == 0])
summary(adj)
adj <- adj$coefficients

#Middle of Field Efficiency
middle <- data[season > 2016 & air_yards > 4 & air_yards < 21 &
                 pass_location == "middle", 
               .(passer_player_name, air_yards, cpoe, epa, 
                 incomplete_pass, interception, posteam)]
middle <- middle[, .(avg_air_yards = sum(air_yards) / .N, 
                     avg_cpoe = sum(cpoe, na.rm = T) / .N, 
                     avg_cp = 1 - sum(incomplete_pass) / .N,
                     int_rate = sum(interception) / .N,
                     avg_epa = sum(epa) / .N,
                     throws = .N), by = "passer_player_name"][throws > 99]
tmp <- data[season > 2016, .N, by = "passer_player_name"]
middle <- merge(middle , tmp, all.x = T, by = "passer_player_name")
middle[, pct_throws := throws / N]
tmp <- data[season > 2016, .N, by = c("passer_player_name", "posteam")]
tmp <- tmp[, max := max(N), by = "passer_player_name"][N == max]
middle <- merge(middle, tmp[, c(1,2)], all.x = T, by = "passer_player_name")

setnames(middle, "posteam", "team_abbr")
tmp <- as.data.table(teams_colors_logos)
middle <- merge(middle, tmp[, c("team_abbr", "team_color", "team_color2")], 
                by = "team_abbr", all.x = T)
tmp <- unique(middle[, c("team_color", "team_abbr")])
col <- tmp[, team_color]
names(col) <- tmp[, team_abbr]

ggplot(middle[passer_player_name != "J.Brissett"], 
       aes(avg_epa, avg_cpoe, label = passer_player_name)) +
  geom_point(aes(color = team_abbr, size = pct_throws, alpha = 0.5), 
             show.legend = F) + 
  scale_color_manual(values = col) +
  scale_size_continuous(range = c(2,8)) +
  geom_text_repel(max.overlaps = 15) +
  geom_vline(xintercept = middle[passer_player_name != "J.Brissett", 
                                 mean(avg_epa)], 
             linetype = "dashed", color = "red") +
  geom_hline(yintercept = middle[passer_player_name != "J.Brissett", 
                                 mean(avg_cp)], 
             linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(x = "EPA_per_Play", y = "CPOE", 
       title = "How good are QBs at throwing over the middle of the field?", 
       subtitle = "Throws between 5 and 20 Yards downfield; Circle size represents percentage of total attempts") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(plot.subtitle = element_text(hjust = 0.5))

data[play_type == "pass" & week < 19, sum(epa, na.rm = T) / .N, 
     by = "posteam"][order(V1, decreasing = T)]

#49ers QB Efficency
niners <- data[posteam == "SF" & play_type %in% c("run", "pass"), 
               .(season, week, play_type, epa, passer_player_name, down, 
                 yardline_100, ydstogo, yards_gained, fumble_lost)]
niners[, first_down := fifelse(yards_gained >= ydstogo, 1, 0)]

niners[fumble_lost == 1 & play_type == "pass" & yards_gained > 0, 
       adj_epa := adj[1] + down * adj[2] + yardline_100 * adj[3] +
         ydstogo * adj[4] + yards_gained * adj[5] + first_down * adj[6]]
niners[, adj_epa := fifelse(is.na(adj_epa), epa, adj_epa)]

tmp <- niners[, .N, by = c("season", "week", "passer_player_name")]
tmp <- na.omit(tmp)
tmp[, plays := sum(N), by = c("season", "week")]
tmp[, ratio := N / plays]
tmp <- tmp[ratio > 0.8]
setnames(tmp, "passer_player_name", "QB")
niners <- merge(niners, tmp[, c(1:3)], by = c("season", "week"), all.x = T)
niners[, QB := fifelse(is.na(QB), "Multiple QBs", QB)]

niners[play_type == "pass", epa_pass := sum(adj_epa) / .N,
       by = c("season", "week")]
niners[play_type == "pass" & passer_player_name == "J.Garoppolo", 
       qb_epa_pass := sum(adj_epa) / .N]
niners[, epa_total := sum(epa) / .N, by = c("season", "week")]
niners[QB == "J.Garoppolo", qb_epa_total := sum(epa) / .N]
niners <- unique(niners[!is.na(epa_pass)], by = c("season", "week"))
niners[season == 2019 & week == 20, game := "NFC Championship 2019"]
niners[season == 2018 & week == 9, game := "Raiders 2018"]

line <- lm(epa_total ~ epa_pass, data = niners)

ggplot(niners, aes(epa_pass, epa_total, color = QB)) +
  geom_point(size = 4, alpha = 0.6) +
  scale_color_manual(values = c("B.Hoyer" = "darkgrey", 
                                "C.Beathard" = "orange", 
                                "J.Garoppolo" = "#aa0000", 
                                "N.Mullens" = "dodgerblue", 
                                "T.Lance" = "green", 
                                "Multiple QBs" = "black")) +
  geom_text(aes(label = game), nudge_y = 0.025, size = 3.5) +
  geom_vline(xintercept = niners[passer_player_name == "J.Garoppolo", 
                                 qb_epa_pass], 
             linetype = "dashed", color = "#aa0000") +
  geom_hline(yintercept = niners[QB == "J.Garoppolo", qb_epa_total], 
             linetype = "dashed", color = "#aa0000") +
  geom_abline(slope = coef(line)[["epa_pass"]], 
              intercept = coef(line)[["(Intercept)"]]) +
  theme_minimal() + 
  labs(title = "49ers Offensive Efficiency by QB since 2017",
       subtitle = "Dashed lines equal Garoppolos averages | Deviations from trend line equal rushing performance", 
       x = "Pass EPA per Play", y = "Total EPA per Play") + 
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(plot.subtitle = element_text(hjust = 0.5))

#First Quarter Scores
test <- data[qtr == 1 & half_seconds_remaining == 900]
test[, score := posteam_score + defteam_score]
test <- test[, c("score", "total_line")]
test <- na.omit(test)

ggplot(test[score < 25], aes(score)) +
  geom_histogram() +
  theme_minimal()

ggplot(test, aes(total_line, score)) +
  geom_point(alpha = 0.25, size = 2) +
  geom_smooth(method = "lm", formula = y ~ x) +
  theme_minimal() +
  labs(title = "First quarter score predicted by total line", 
       x = "Total Line", y = "Score") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(plot.subtitle = element_text(hjust = 0.5))

summary(lm(score ~ total_line, data = test))

#Expected Series Conversion Rate on Ints Past Sticks
x <- c("wpa", "epa", "interception", "passer_player_name", "yardline_100", 
       "down", "ydstogo", "qb_dropback", "air_yards", "posteam")
qbs <- data[season > 2017 & season < 2021, ..x]
qbs <- na.omit(qbs)

sc <- data[, .(conv_rate = sum(series_success) / .N, sample = .N), 
           by = c("down", "ydstogo")]
sc <- na.omit(sc)
sc[sample < 5, conv_rate := 0]

qb_teams <- unique(data[season == 2020, c("passer_player_name", "posteam")])
qb_teams <- na.omit(qb_teams)
setnames(qb_teams, "posteam", "team_abbr")
tmp <- as.data.table(teams_colors_logos)
qb_teams <- merge(qb_teams, tmp[, c("team_abbr", "team_color")], 
                  by = "team_abbr", all.x = T)

tmp <- qbs[, sum(qb_dropback), by = "passer_player_name"][V1 > 500]
qbs <- qbs[passer_player_name %in% tmp[, passer_player_name]]
qbs <- merge(qbs, sc, by = c("down", "ydstogo"), all.x = T)
qbs[, past_sticks := fifelse(air_yards >= ydstogo, 1, 0)]
lost <- qbs[interception == 1, .(avg_ps_rate = sum(past_sticks) / .N, 
                                 avg_conv_rate = sum(conv_rate) / .N, 
                                 interceptions = sum(interception)), 
            by = "passer_player_name"]
lost <- merge(lost, qb_teams, by = "passer_player_name", all.x = T)
lost <- lost[-c(24)]
lost[passer_player_name == "A.Luck", ":=" (team_abbr = "IND", 
                                           team_color = "#002c5f")]
lost[passer_player_name == "E.Manning", ":=" (team_abbr = "NYG", 
                                              team_color = "#0b2265")]
lost[passer_player_name == "J.Rosen", ":=" (team_abbr = "ARI", 
                                            team_color = "#97233f")]

tmp <- unique(lost[, c("team_color", "team_abbr")])
col <- tmp[, team_color]
names(col) <- tmp[, team_abbr]

ggplot(lost, aes(avg_conv_rate, avg_ps_rate, label = passer_player_name)) +
  geom_point(aes(color = team_abbr), size = 2) +
  scale_color_manual(values = col) +
  geom_text_repel() +
  geom_vline(xintercept = qbs[interception == 1, sum(conv_rate) / .N], 
             linetype = "dashed") +
  geom_hline(yintercept = qbs[interception == 1, sum(past_sticks) / .N], 
             linetype = "dashed") +
  theme_minimal() +
  theme(legend.position = "none") +
  labs(title = "Some Title", 
       subtitle = "Min. 500 Dropbacks, 2018-2020", 
       x = "Avg expected series conversion rate on intercepted passes",
       y = "Rate of interceptions that went past the sticks") +
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(plot.subtitle = element_text(hjust = 0.5))
