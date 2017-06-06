\LoadClass[12pt,cap,c5size,UTF8]{ctexart}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{PatternRecognition}[2016/10/08 PR class Latex Template written by Cosmo]
\RequirePackage{graphicx, fancyhdr,amsfonts, amsmath, amsopn,amssymb, amstext, amsthm, bm, delarray, ,eqnarray, theorem, multicol, booktabs}

\renewcommand{\title}[1]{
	\center{\heiti \zihao{3} #1}
}

%\titleformat{\subsection}
%{\zihao{4}\heiti\raggedright}
%{}{0em}
%{}
%
%\titleformat{\subsubsection}
%{\zihao{-4}\heiti\raggedright}
%{}{0em}
%{}

\newcommand{\ctitle}[1]{\begin{center}\zihao{3}\heiti #1\end{center}}

\newcommand{\cauthorinfo}[5]{\begin{center}\zihao{5}\fangsong#1 \\ \zihao{-5}\songti {#2} #3 \\ \zihao{-5}\songti #4 #5\end{center}}

\newcommand{\cabstract}[1]{
	\begin{center}
		\zihao{-4} \heiti 摘要
	\end{center}
	\quad \quad#1
}
\newcommand{\etitle}[1]{\begin{center}\zihao{3}\heiti #1\end{center}}

\newcommand{\eauthorinfo}[5]{\begin{center}\zihao{4}\fangsong#1 \\ \zihao{-5}\songti {#2} #3 \\ \zihao{-5}\songti #4 #5\end{center}}

\newcommand{\eabstract}[1]{
	\begin{center}
		\zihao{-4} \heiti Abstract
	\end{center}
	\quad \quad#1
}

\newcommand{\pythonCode}[1]{
\begin{minted}
[
frame=lines,
framesep=2mm,
baselinestretch=1.2,
fontsize=\footnotesize,
linenos
]
{python}
#1
\end{minted}
}
