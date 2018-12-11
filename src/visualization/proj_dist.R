library(readr)
library(ggplot2)

# degree distribution of projections
uu <- read_tsv('src/data/processed/uu_full_headers.csv')
u_edges <- c(uu$e1,uu$e2)
ut <- table(u_edges)
ut_df <- data.frame(ut)
utf <- data.frame(table(ut_df$Freq))
utf$size = as.numeric(as.character(utf$Var1))
up_df <- data.frame(Freq = utf$Freq, size=utf$size)
up_df$cat = 'User'
up_df$graph = 'Projection'

aa <- read_tsv("src/data/processed/aa_full_headers.csv")
a_edges <- c(aa$e1, aa$e2)
at_df <- data.frame(table(a_edges))
atf <- data.frame(table(at_df$Freq))
atf$size = as.numeric(as.character(atf$Var1))
ap_df <- data.frame(Freq = atf$Freq, size=atf$size)
ap_df$cat = 'Article'
ap_df$graph = 'Projection'

u_orig <- read_csv('src/data/processed/user_counts.csv')
a_orig <- read_csv('src/data/processed/article_counts.csv')
u_df <- data.frame(Freq = u_orig$users, size=u_orig$num_edits)
a_df <- data.frame(Freq = a_orig$articles, size=a_orig$num_edits)
u_df$cat = 'User'
u_df$graph = 'Bipartite'
a_df$cat = 'Article'
a_df$graph = 'Bipartite'

all_df <- rbind(up_df, ap_df, u_df, a_df)

ggplot(all_df, aes(x=size, y=Freq, group=graph, col=graph)) + geom_point(size=0.1) +
  scale_x_log10() + scale_y_log10() +
  facet_wrap(~cat) +
  xlab("Node Degree") + 
  ylab("Count") +
  theme(legend.position = 'bottom') 
ggsave('proj_vs_bipartite.pdf')
