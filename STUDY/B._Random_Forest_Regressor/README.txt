B. Random Forest Regressor
Не решена
Ограничение времени	10 секунд
Ограничение памяти	512 Мб
Ввод	стандартный ввод или input.txt
Вывод	стандартный вывод или output.txt
\section*{Задача: Random Forest (Regressor)}

Вам предоставлен датасет:

\texttt{Salaries.csv}

   Id        EmployeeName                                        JobTitle    BasePay  OvertimePay   OtherPay  Benefits   TotalPay  TotalPayBenefits  Year  Notes         Agency  Status
0   1      NATHANIEL FORD  GENERAL MANAGER-METROPOLITAN TRANSIT AUTHORITY  167411.18         0.00  400184.25       NaN  567595.43         567595.43  2011    NaN  San Francisco     NaN
1   2        GARY JIMENEZ                 CAPTAIN III (POLICE DEPARTMENT)  155966.02    245131.88  137811.38       NaN  538909.28         538909.28  2011    NaN  San Francisco     NaN
2   3      ALBERT PARDINI                 CAPTAIN III (POLICE DEPARTMENT)  212739.13    106088.18   16452.60       NaN  335279.91         335279.91  2011    NaN  San Francisco     NaN
3   4   CHRISTOPHER CHONG            WIRE ROPE CABLE MAINTENANCE MECHANIC   77916.00     56120.71  198306.90       NaN  332343.61         332343.61  2011    NaN  San Francisco     NaN
4   5     PATRICK GARDNER    DEPUTY CHIEF OF DEPARTMENT,(FIRE DEPARTMENT)  134401.60      9737.00  182234.59       NaN  326373.19         326373.19  2011    NaN  San Francisco     NaN
5   6      DAVID SULLIVAN                       ASSISTANT DEPUTY CHIEF II  118602.00      8601.00  189082.74       NaN  316285.74         316285.74  2011    NaN  San Francisco     NaN
6   7           ALSON LEE              BATTALION CHIEF, (FIRE DEPARTMENT)   92492.01     89062.90  134426.14       NaN  315981.05         315981.05  2011    NaN  San Francisco     NaN
7   8       DAVID KUSHNER                  DEPUTY DIRECTOR OF INVESTMENTS  256576.96         0.00   51322.50       NaN  307899.46         307899.46  2011    NaN  San Francisco     NaN
8   9      MICHAEL MORRIS              BATTALION CHIEF, (FIRE DEPARTMENT)  176932.64     86362.68   40132.23       NaN  303427.55         303427.55  2011    NaN  San Francisco     NaN
9  10  JOANNE HAYES-WHITE          CHIEF OF DEPARTMENT, (FIRE DEPARTMENT)  285262.00         0.00   17115.73       NaN  302377.73         302377.73  2011    NaN  San Francisco     NaN
1e-05

\subsection*{Требуется:}

\begin{enumerate} \item Загрузить данные из файла. \item Удалить выбросы и дубликаты. \item Закодировать признаки с типом \texttt{object}. \item Разделить данные на признаки и целевую переменную \texttt{salary_in_usd}. \item Разделить данные на обучающую и тестовую выборки. \item Создать модель \texttt{RandomForestRegressor} с параметром: [ n_estimators = 10 ] \item Обучить модель. \item Найти значение OOB-оценки и сохранить его в переменную: [ \texttt{oob_score} ] \end{enumerate}

\subsection*{Критерий прохождения задания:}

[ \texttt{oob_score} \geq 0.8 ]

\subsection*{Формат ответа:}

В качестве ответа необходимо вывести значение переменной:

[ \texttt{oob_score} ]
