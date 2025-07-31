DELIMITER //

DROP TRIGGER IF EXISTS position_existance_trigger; //

CREATE TRIGGER position_existance_trigger
BEFORE INSERT ON PlayerPositions
FOR EACH ROW
BEGIN
    DECLARE m_count INT;
    SELECT COUNT(*) INTO m_count
    FROM Position
    WHERE position_ID = NEW.position;

    IF m_count = 0 THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Position does not exist!';
    END IF;
END; //
