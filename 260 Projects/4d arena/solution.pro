% ahmet firat gamsiz
% 2020400180
% compiling: yes
% complete: yes

distance(0, 0, 0).  % a dummy predicate to make the sim work.

% return the travel cost of an agent
travel_cost(Agent, Cost):-Agent.class = wizard, Cost is 2.
travel_cost(Agent, Cost):-Cost is 5.

%calculate manhattan distance between two agents
distance(Agent, TargetAgent, Distance):- 
    Distance is abs(Agent.x - TargetAgent.x) + abs(Agent.y - TargetAgent.y).

%calculate manhattan distance and universal distance between two agents
multiverse_distance(StateId, AgentId, TargetStateId, TargetAgentId, Distance):-
    state(StateId, Agent,_,_), 
    state(TargetStateId, TargetAgent,_,_),
    history(StateId,UniverseId,Time,_),
    history(TargetStateId,TargetUniverseId,TargetTime,_),
    distance(Agent.AgentId, TargetAgent.TargetAgentId,EucDistance),
    travel_cost(Agent.AgentId, Cost),
    Distance is EucDistance + Cost*(abs(Time-TargetTime)+abs(UniverseId-TargetUniverseId)). %eucdistance is actually manhattan distance

%predicates to find the nearest agent in a dictionary
find_min(Agent, TargetAgentDict, [H], H):-
    TargetAgentDict.H.name \= Agent.name.

find_min(Agent, TargetAgentDict, [H,N|T],M):-
    Agent.name = TargetAgentDict.H.name,
    find_min(Agent, TargetAgentDict, [N|T],M).

find_min(Agent, TargetAgentDict, [H,N|T],M):-
    Agent.name = TargetAgentDict.N.name,
    find_min(Agent, TargetAgentDict, [H|T],M).

find_min(Agent, TargetAgentDict, [H,N|T],M):-
    distance(Agent, TargetAgentDict.H,A),
    distance(Agent,TargetAgentDict.N,B), 
    A=<B,
    find_min(Agent, TargetAgentDict, [H|T],M).

find_min(Agent, TargetAgentDict, [H,N|T],M):-
    distance(Agent, TargetAgentDict.H,A),
    distance(Agent, TargetAgentDict.N,B), 
    A>B,
    find_min(Agent, TargetAgentDict, [N|T],M).

% return the nearest agent in the same state
nearest_agent(StateId, AgentId, NearestAgentId, Distance):- 
    state(StateId, Agent,_,_),
    state(StateId, TargetAgentDict,_,TargetAgentIds),
    find_min(Agent.AgentId, TargetAgentDict, TargetAgentIds, NearestAgentId),
    distance(Agent.AgentId, TargetAgentDict.NearestAgentId, Distance), NearestAgentId is NearestAgentId.

% return the nearest agent in the multiverse
% calculate all distances and sort them, then return the first one
nearest_agent_in_multiverse(StateId, AgentId, TargetStateId, TargetAgentId, Distance):-
    state(StateId, Agent,_,_),
    findall(TargetStateId, state(TargetStateId, _,_,_), TargetStates),
    length(TargetStates, NumStates),
    NumStates > 0,
    findall([Distance, TargetStateId, TargetAgentId], 
        (member(TargetStateId, TargetStates),
        state(TargetStateId, TargetAgentDict,_,TargetAgentIds),
    TargetAgentDict.TargetAgentId.name \= Agent.AgentId.name,
    multiverse_distance(StateId, AgentId, TargetStateId, TargetAgentId, Distance)), Distances),
    sort(1, @=<, Distances, Sorted),
    Sorted = [[Distance, TargetStateId, TargetAgentId]|_].

% count the number of agents in a state except the agent itself or its clones
num_agents_in_state(StateId, Name, NumWarriors, NumWizards, NumRogues):-
    state(StateId, AgentDict, _, AgentIds),
    findall(AgentId, (member(AgentId, AgentIds), AgentDict.AgentId.class = warrior, AgentDict.AgentId.name\=Name), WarriorIds),
    length(WarriorIds, NumWarriors),
    findall(AgentId, (member(AgentId, AgentIds), AgentDict.AgentId.class = wizard, AgentDict.AgentId.name\=Name), WizardIds),
    length(WizardIds, NumWizards),
    findall(AgentId, (member(AgentId, AgentIds), AgentDict.AgentId.class = rogue, AgentDict.AgentId.name\=Name), RogueIds),
    length(RogueIds, NumRogues).

% calculate the difficulty of a state
difficulty_of_state(StateId, Name, AgentClass, Difficulty):-
    num_agents_in_state(StateId, Name, NumWarriors, NumWizards, NumRogues),
    ((AgentClass = warrior, Difficulty is 5*NumWarriors + 8*NumWizards + 2*NumRogues);
    (AgentClass = wizard, Difficulty is 2*NumWarriors + 5*NumWizards + 8*NumRogues);
    (AgentClass = rogue, Difficulty is 8*NumWarriors + 2*NumWizards + 5*NumRogues)).

% conditions for portaling to a state (these were given in the simulator already)
portal_conditions(StateId, AgentId, TargetStateId):-
    state(StateId, Agents, CurrentTurn, TurnOrder),
    history(StateId, UniverseId, Time, Turn),
    Agent = Agents.get(AgentId),

    global_universe_id(GlobalUniverseId),
    universe_limit(UniverseLimit),
    GlobalUniverseId < UniverseLimit,

    history(TargetStateId, TargetUniverseId, TargetTime, TargetTurn),

    length(TurnOrder, NumAgents),
    NumAgents > 1,

    current_time(TargetUniverseId, TargetUniCurrentTime, _),
    TargetTime < TargetUniCurrentTime,

    (Agent.class = wizard -> TravelCost = 2; TravelCost = 5),
    Cost is abs(TargetTime - Time)*TravelCost + abs(TargetUniverseId - UniverseId)*TravelCost,
    Agent.mana >= Cost,

    get_earliest_target_state(TargetUniverseId, TargetTime, TargetStateId),

    state(TargetStateId, TargetAgents, _, TargetTurnOrder),
    TargetState = state(TargetStateId, TargetAgents, _, TargetTurnOrder),

    \+tile_occupied(Agent.x, Agent.y, TargetState).

% conditions for portaling to now, to a state (these were given in the simulator already)
portal_to_now_conditions(StateId, AgentId, TargetStateId):-
    state(StateId, Agents, CurrentTurn, TurnOrder),
    history(StateId, UniverseId, Time, Turn),
    State = state(StateId, Agents, CurrentTurn, TurnOrder),
    Agent = Agents.get(AgentId),

    length(TurnOrder, NumAgents),
    NumAgents > 1,
    
    history(TargetStateId, TargetUniverseId, TargetTime, TargetTurn),
    current_time(TargetUniverseId, TargetTime, 0),
    \+(TargetUniverseId = UniverseId),

    (Agent.class = wizard -> TravelCost = 2; TravelCost = 5),
    Cost is abs(TargetTime - Time)*TravelCost + abs(TargetUniverseId - 
        UniverseId)*TravelCost,
    Agent.mana >= Cost,

    get_latest_target_state(TargetUniverseId, TargetTime, TargetStateId),
    state(TargetStateId, TargetAgents, _, TargetTurnOrder),
    TargetState = state(TargetStateId, TargetAgents, _, TargetTurnOrder),
    \+tile_occupied(Agent.x, Agent.y, TargetState).

% given state is assumed to be traversable
traversable_state(StateId, AgentId, StateId).

% predicates to check a state is traversable
traversable_state(StateId, AgentId, TargetStateId):-
    portal_conditions(StateId, AgentId, TargetStateId),!.

traversable_state(StateId, AgentId, TargetStateId):-
    portal_to_now_conditions(StateId, AgentId, TargetStateId).

% find all traversable states return easiest one
easiest_traversable_state(StateId, AgentId, TargetStateId):-
    state(StateId, AgentDict, _, _),
    Agent = AgentDict.AgentId,
    Name = Agent.name,
    Class = Agent.class,

    findall(TargetStateId, traversable_state(StateId, AgentId, TargetStateId), TargetStateIds),
    findall([Difficulty,TargetStateId], (member(TargetStateId,TargetStateIds), difficulty_of_state(TargetStateId, Name, Class, Difficulty), Difficulty>0), Difficulties),
    sort(1,@<,Difficulties, SortedDifficulties),
    [[Difficulty,TargetStateId]|_] = SortedDifficulties.

% check the target is in attack range
in_attack_range(Agent, TargetAgent):-
    ((Agent.class = warrior, Range = 1);
    (Agent.class = wizard, Range = 10);
    (Agent.class = rogue, Range = 5)),
    distance(Agent, TargetAgent, Distance),!,
    Distance =< Range.

% portal to now policy
policy(StateId, AgentId, Action):-
    easiest_traversable_state(StateId, AgentId, TargetStateId),
    portal_to_now_conditions(StateId, AgentId, TargetStateId),!,
    history(TargetStateId, TargetUniverseId,_,_),
    Action = [portal_to_now,TargetUniverseId].

% portal policy
policy(StateId, AgentId, Action):-
    easiest_traversable_state(StateId, AgentId, TargetStateId),
    portal_conditions(StateId, AgentId, TargetStateId),!,
    history(TargetStateId, TargetUniverseId, TargetTime,_),
    Action = [portal,TargetUniverseId, TargetTime].

% attack policy
policy(StateId, AgentId, Action):-
    nearest_agent(StateId, AgentId, NearestAgentId, Distance),
    state(StateId, AgentDict, _, _),
    Agent = AgentDict.AgentId,
    TargetAgent = AgentDict.NearestAgentId,
    in_attack_range(Agent, TargetAgent),!,
    ((Agent.class = warrior, Action = [melee_attack, NearestAgentId]);
    (Agent.class = wizard, Action = [magic_missile, NearestAgentId]);
    (Agent.class = rogue, Action = [ranged_attack, NearestAgentId])).

% move policy
% find nearest agent, check the way is clear, move towards it
policy(StateId, AgentId, Action):-
    nearest_agent(StateId, AgentId, NearestAgentId, Distance),
    state(StateId, AgentDict, _, _),
    State = state(StateId, AgentDict, _, _),
    Agent = AgentDict.AgentId,
    TargetAgent = AgentDict.NearestAgentId,
    ((Agent.x < TargetAgent.x,
        X is Agent.x + 1,
        \+ tile_occupied(X, Agent.y, State), 
        Action = [move_right]);
    (Agent.y < TargetAgent.y,
        Y is Agent.y + 1,
        \+ tile_occupied(Agent.x, Y, State), 
        Action = [move_up]);
    (Agent.x > TargetAgent.x,
        X is Agent.x - 1,
        \+ tile_occupied(X, Agent.y, State),
        Action = [move_left]);
    (Agent.y > TargetAgent.y,
        Y is Agent.y - 1,
        \+ tile_occupied(Agent.x, Y, State),
        Action = [move_down])),!.

% rest policy
policy(StateId, AgentId, Action):-
    Action = [rest].

% find appropriate policy
basic_action_policy(StateId, AgentId, Action):- 
    policy(StateId, AgentId, Action).


