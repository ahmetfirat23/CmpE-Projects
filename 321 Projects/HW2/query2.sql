SELECT session_id, assigned_jury_username, rating, DATE_FORMAT(STR_TO_DATE(date, '%d.%m.%Y'), "%d/%m/%Y") AS date
FROM MatchSession AS M
WHERE STR_TO_DATE(date, '%d.%m.%Y') < "20240101"
ORDER BY STR_TO_DATE(date, '%d.%m.%Y') ASC;
