CREATE TABLE User (
    username CHAR(50),
    password CHAR(50) NOT NULL,
    name CHAR(50),
    surname CHAR(50),
    PRIMARY KEY (username)
);

CREATE TABLE Player (
    username CHAR(50),
    date_of_birth DATE,
    height INT,
    weight INT,
    PRIMARY KEY (username),
    FOREIGN KEY (username) REFERENCES User (username)
);

CREATE TABLE Coach (
    username CHAR(50),
    nationality CHAR(50) NOT NULL,
    PRIMARY KEY (username),
    FOREIGN KEY (username) REFERENCES User (username)
);

CREATE TABLE Jury (
    username CHAR(50),
    nationality CHAR(50) NOT NULL,
    PRIMARY KEY (username),
    FOREIGN KEY (username) REFERENCES User (username)
);

CREATE TABLE Position (
    position_ID INT,
    position_name CHAR(50) NOT NULL,
    PRIMARY KEY (position_ID)
);

CREATE TABLE Channel (
    channel_ID INT,
    channel_name CHAR(50) NOT NULL,
    PRIMARY KEY (channel_ID)
);

CREATE TABLE Team (
    team_ID INT,
    team_name CHAR(50) NOT NULL,
    channel_ID INT NOT NULL,
    PRIMARY KEY (team_ID),
    FOREIGN KEY (channel_ID) REFERENCES Channel (channel_ID)
);

CREATE TABLE Stadium (
    stadium_id INT,
    stadium_name CHAR(50) NOT NULL,
    stadium_country CHAR(50) NOT NULL,
    PRIMARY KEY (stadium_id)
);

CREATE TABLE Date_Time (
    date_of DATE,
    timeslot INT,
    PRIMARY KEY (date_of, timeslot),
    CHECK (timeslot >= 1 AND timeslot <= 3)
);

CREATE TABLE Match_Sessions (
    session_ID INT,
    team_ID INT NOT NULL,
    stadium_id INT NOT NULL,
    date_of DATE NOT NULL,
    timeslot INT NOT NULL,
    PRIMARY KEY (session_ID),
    UNIQUE (team_ID, date_of, timeslot),
    UNIQUE (stadium_id, date_of, timeslot),
    FOREIGN KEY (team_ID) REFERENCES Team (team_ID),
    FOREIGN KEY (stadium_id) REFERENCES Stadium (stadium_id),
    FOREIGN KEY (date_of, timeslot) REFERENCES Date_Time (date_of, timeslot)
);

CREATE TABLE Can_Play (
    username CHAR(50),
    position_id INT,
    PRIMARY KEY (username, position_id),
    FOREIGN KEY (username) REFERENCES Player (username),
    FOREIGN KEY (position_id) REFERENCES Position (position_ID)
);

CREATE TABLE Registered (
    username CHAR(50),
    team_ID INT,
    PRIMARY KEY (username, team_ID),
    FOREIGN KEY (username) REFERENCES Player (username),
    FOREIGN KEY (team_ID) REFERENCES Team (team_ID)
);

CREATE TABLE In_Contract (
    username CHAR(50),
    team_ID INT NOT NULL UNIQUE,
    contract_start DATE NOT NULL,
    contract_finish DATE NOT NULL,
    PRIMARY KEY (username),
    FOREIGN KEY (username) REFERENCES Coach (username),
    FOREIGN KEY (team_ID) REFERENCES Team (team_ID)
);

CREATE TABLE Rates (
    session_ID INT,
    username CHAR(50),
    rating DECIMAL(1, 1) NOT NULL,
    PRIMARY KEY (session_ID),
    FOREIGN KEY (session_ID) REFERENCES Match_Sessions (session_ID),
    FOREIGN KEY (username) REFERENCES Jury (username)
);

CREATE TABLE Player_Plays_In (
    username CHAR(50),
    session_ID INT,
    position_ID INT NOT NULL,
    PRIMARY KEY (username, session_ID),
    FOREIGN KEY (username) REFERENCES Player (username),
    FOREIGN KEY (session_ID) REFERENCES Match_Sessions (session_ID),
    FOREIGN KEY (position_ID) REFERENCES Position (position_ID)
);
