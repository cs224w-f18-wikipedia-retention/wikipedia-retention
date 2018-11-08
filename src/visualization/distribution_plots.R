# build log-log graph of different graph properties
library(readr)
library(ggplot2)
ac = read_csv("article_counts.csv")
uc = read_csv("user_counts.csv")
wc = read_csv("word_counts.csv")
uwc = read_csv("user_word_counts.csv")

# combine all into one df
colnames(ac) <- c("x","y")
colnames(uc) <- c("x","y")
colnames(wc) <- c("x","y")
colnames(uwc) <- c("x","y")
ac$title = 'Article Edits'
uc$title = 'User Edits'
wc$title = 'Edit Words'
uwc$title = 'Total User Words'
dat = rbind(ac,uc,wc,uwc)

# plot segments
ggplot(dat, aes(x=x, y=y, group=title)) + geom_point(size=0.1) +
  scale_x_log10() + scale_y_log10() +
  ylab("count") +
  facet_wrap(~title)

ggsave("dist_segment.pdf")

# plot all on one
ggplot(dat, aes(x=x, y=y, col=title)) + geom_point(size=0.1) +
  scale_x_log10() + scale_y_log10() +
  ylab("count") +
  theme(legend.position = 'bottom') +
  theme(plot.title = element_text(hjust = 0.5),
        legend.background = element_rect(fill="gray90", size=.5, linetype="dotted"))

ggsave("dist_all.pdf")

# part 2 - plot distribution of account durations
ad = read_csv('account_durations.csv')

# plot dist
ggplot(ad, aes(x=days, y=num_users)) + geom_point(size=0.1) +
    scale_y_log10()

ggsave("account_duration_nologx.pdf")

# alt: scale x
ggplot(ad, aes(x=days+1, y=num_users)) + geom_point(size=0.1) +
  scale_x_log10() + scale_y_log10()

ggsave("account_duration.pdf")

# part 3 - plot distribution of feature improvement to logloss
uafr = read_csv('user_article_feature_ranking.csv')
ggplot(uafr, aes(x=fnum, y=logloss)) + geom_line() +
  xlab("Top Features Used") + ylab("Log Loss")

ggsave("numfeat_vs_logloss.pdf")

# rank vs coef (not using atm)
ggplot(uafr, aes(x=rank, y=coef)) + geom_point() +
  xlab("Feature Rank") + ylab("Optimal Model Weight")

ggsave("rank_vs_coef.pdf")

# part 4 - plot feature histogram
py <- read_csv("preds.csv")
py$label = "Left Wikipedia"
py$label[py$`# y` == 1] = "Joined Wikipedia"
ggplot(py,aes(x=pred, group=label, fill=label)) + stat_bin(bins=20,aes(y=..density..),position="dodge") +
  scale_fill_grey() +
  theme(legend.position = 'bottom', legend.title = element_blank())

ggsave("pred_density_split.pdf")

# role plots
role_new <- read.csv("enwiki-projection-user-roles-2007-1.csv",header=FALSE, sep='\t')
role_new <- role_new[2:length(role_new[,1]),] # strip bad first row
uu_proj_edges <- read.csv("enwiki-projection-user-2007-1.csv",header=FALSE, sep='\t')

# role histogram
role_new$role = as.integer(role_new$V2)
ggplot(role_new, aes(x=role)) + geom_histogram(bins=length(unique(role_new$role)))
ggsave("uu_proj_roles.pdf")

# degree distribution of projected graph
upt <- table(uu_proj_edges[,1])
upt_df <- data.frame(upt)
uptf_df <- data.frame(table(upt_df[,2]))
uptf_df$Degree = as.numeric(uptf_df$Var11)
ggplot(uptf_df, aes(x=as.numeric(Var1), y=Freq)) + geom_point(size=0.1) +
  scale_x_log10() + scale_y_log10() + 
  xlab("Degree") + ylab("Node Count")

ggsave("user_projection_degree_dist.pdf")
