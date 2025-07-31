SELECT C.name, C.surname
FROM Coach C
WHERE 2 <= (SELECT COUNT(*)
			FROM MatchSession M, Team T
            WHERE M.team_ID = T.team_ID AND T.coach_username = C.username)
ORDER BY C.surname DESC;