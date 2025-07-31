DELIMITER //

DROP TRIGGER IF EXISTS check_overlap; //

CREATE TRIGGER check_overlap
BEFORE INSERT ON MatchSession 
FOR EACH ROW
BEGIN
	DECLARE m_count INT;
    
    SELECT COUNT(*) INTO m_count
    FROM MatchSession
    WHERE (time_slot = NEW.time_slot OR
    time_slot = NEW.time_slot + 1 OR
    time_slot = NEW.time_slot - 1) AND
    stadium_ID = NEW.stadium_ID AND 
    date = NEW.date;
    
	IF m_count > 0 THEN
        SIGNAL SQLSTATE '45000' 
        SET MESSAGE_TEXT = 'Overlapping session';
    END IF;
END;
    
    