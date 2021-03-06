\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{snamc2013}[2012/11/10]

\LoadClass[twocolumn,10pt]{article}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Page layout

\RequirePackage[width=7in,top=1in,bottom=1in,centering,a4paper]{geometry} % texlive-latex-base
% \setlength{\headsep}{0pt}
\setlength{\headheight}{18pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Headers and footers

\RequirePackage{fancyhdr} % texlive-latex-base
\fancyhf{}
\pagestyle{fancy}
\fancypagestyle{plain}{
  \fancyhf{}
  \lhead{\footnotesize\emph{Joint International Conference on Supercomputing in
      Nuclear Applications and Monte Carlo 2013 (SNA + MC 2013)} \\ La Cit\'{e}
    des Sciences et de l'Industrie, Paris, France, October 27--31, 2013}
  \lfoot{}
}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title

% Defines 'strip' environment for single column
\RequirePackage{cuted}   % texlive-latex-extra

% Change style of title
\RequirePackage{titling} % texlive-latex-extra
\pretitle{\begin{center}\Large\bfseries}
\posttitle{\end{center}\vspace{6pt}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author block

\RequirePackage{authblk} % texlive-latex-extra
\renewcommand\Authfont{\normalsize} % 10pt
\renewcommand\Affilfont{\small\itshape} % 9pt, italic
\date{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Abstract

% Provide \abstract and \keywords commands
\renewcommand{\abstract}[1]{\gdef\@abstract{#1}}
\newcommand{\keywords}[1]{\gdef\@keywords{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Section headings

\RequirePackage{titlesec} % texlive-latex-extra
\titlelabel{\thetitle. }
\titleformat*{\section}{\large\bfseries}
\titleformat*{\subsection}{\normalsize\bfseries}
\titleformat*{\subsubsection}{\normalsize\bfseries}
\renewcommand{\thesection}{\Roman{section}}
\renewcommand{\thesubsection}{\arabic{subsection}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fonts

\usepackage{amsmath} % ams math must be loaded before txfonts
\RequirePackage[T1]{fontenc} % Change font encoding

% Use Times for math font and text font.
\RequirePackage{txfonts} % texlive-fonts-recommended
\usepackage{bm} % bold math must be loaded after Times font

\renewcommand{\footnotesize}{\fontsize{8}{9.6}\selectfont}
\renewcommand{\small}{\fontsize{9}{10.8}\selectfont}
\renewcommand{\normalsize}{\fontsize{10}{12}\selectfont}
\renewcommand{\large}{\fontsize{11}{13.2}\selectfont}
\renewcommand{\Large}{\fontsize{14}{16.8}\selectfont}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Citations and Bibliography

% Use superscript 
\RequirePackage[super]{cite} % texlive-latex-recommended

% Set size of 'References' to 9pt
\renewcommand{\refname}{\normalsize References}

% Use parentheses around citations and make bold
\renewcommand\@citess[1]{\textsuperscript{(\textbf{#1})}}

% Change from [x] to x) in bibliography
\renewcommand\@biblabel[1]{#1)}

% Reduce spacing in bibliography; thanks to Jose Carlos Santos.
\let\@oldthebibliography=\thebibliography
\let\@endoldthebibliography=\endthebibliography
\let\@olditemsep=\itemsep
\renewenvironment{thebibliography}[1]{
  \small
  \begin{@oldthebibliography}{#1}
    \setlength{\itemsep}{-\@olditemsep}
}{
  \@endoldthebibliography
  \end{@oldthebibliography}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figures and Tables

\RequirePackage{caption} % texlive-latex-recommended
\captionsetup{font=small,labelfont=bf,textfont=bf,labelsep=colon}

\setlength{\abovecaptionskip}{0.25\baselineskip}
\setlength{\belowcaptionskip}{0.25\baselineskip}
\setlength{\textfloatsep}{0.5\baselineskip}
\setlength{\intextsep}{0.5\baselineskip}
\renewcommand{\bottomfraction}{0.9}
\renewcommand{\topfraction}{0.9}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Display title, authors, abstract, keywords at beginning

\AtBeginDocument{
  \pagestyle{plain}
  \begin{strip}
    \vspace{-20pt}
    \maketitle

    \vspace{-30pt}
    \addtolength{\leftskip}{0.55in}
    \addtolength{\rightskip}{0.55in}
    \small\@abstract \\

    \textbf{\textit{KEYWORDS}}: \@keywords
    \vspace{6pt}
  \end{strip}
}

