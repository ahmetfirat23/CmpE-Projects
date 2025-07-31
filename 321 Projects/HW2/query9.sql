SELECT name, surname
FROM Player
WHERE height > (SELECT height
				FROM Player
                WHERE TRIM(name) = "Ebrar" AND TRIM(surname) = "Karakurt") AND
	YEAR(STR_TO_DATE(date_of_birth, "%d/%m/%Y")) = (SELECT YEAR(STR_TO_DATE(date_of_birth, "%d/%m/%Y"))
				FROM Player
                WHERE TRIM(name) = "Ebrar" AND TRIM(surname) = "Karakurt")
		

