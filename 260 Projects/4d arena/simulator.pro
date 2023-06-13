print_array([]).
print_array([H|T]) :-
    write(H), nl,
    print_array(T).


tile_occupied(X, Y, State) :-
    width(Width), height(Height),
    (
        (X < 0; Y < 0; X >= Width; Y >= Height);
        (State = state(_, Agents, _, _),
        Agent = Agents.get(_), Agent.x = X, Agent.y = Y)
    ).

can_perform(Class, Action) :-
    (Class = warrior ->
        ActionList = [move_right, move_up, move_left, move_down,
                      portal, portal_to_now, melee_attack, rest],
        member(Action, ActionList));
    (Class = wizard ->
        ActionList = [move_right, move_up, move_left, move_down,
                      portal, portal_to_now, magic_missile, rest],
        member(Action, ActionList));
    (Class = rogue ->
        ActionList = [move_right, move_up, move_left, move_down,
                      portal, portal_to_now, ranged_attack, rest],
        member(Action, ActionList)).

get_newest_state(UniverseId, StateId) :-
    current_time(UniverseId, Time, Turn),
    findall(
        CandidateStateId,
        history(CandidateStateId, UniverseId, Time, Turn),
        CandidateStates
    ),
    max_list(CandidateStates, StateId).

get_earliest_target_state(UniverseId, Time, StateId) :-
    findall(
        CandidateStateId,
        history(CandidateStateId, UniverseId, Time, _),
        CandidateStates
    ),
    min_list(CandidateStates, StateId).

get_latest_target_state(UniverseId, Time, StateId) :-
    findall(
        CandidateStateId,
        history(CandidateStateId, UniverseId, Time, _),
        CandidateStates
    ),
    max_list(CandidateStates, StateId).

get_current_agent_and_state(UniverseId, AgentId, StateId) :-
    get_newest_state(UniverseId, StateId),
    state(StateId, _, CurrentTurn, TurnOrder),
    
    nth0(CurrentTurn, TurnOrder, AgentId).

get_new_time_and_turn(UniverseId, NewTurnOrder, NewTime, NewTurn) :-
    current_time(UniverseId, Time, Turn),
    get_newest_state(UniverseId, StateId),
    state(StateId, _, _, TurnOrder),
    length(TurnOrder, TurnOrderLength),
    length(NewTurnOrder, NewTurnOrderLength),
    (NewTurnOrderLength = TurnOrderLength ->  % if no time travel
        (
            NewTurn is (Turn + 1) mod TurnOrderLength,
            (NewTurn = 0 ->
                NewTime is Time + 1;
                NewTime is Time)
        );
        (
            NewTurn is (Turn) mod NewTurnOrderLength,
            ((\+(Turn = 0), NewTurn = 0) ->
                NewTime is Time + 1;
                NewTime is Time)
        )
    ).

modify_agent(Agent, [], [], Agent).
modify_agent(Agent, Key, Value, NewAgent) :-
    put_dict(Key, Agent, Value, NewAgent).
modify_agent(Agent, [Key|KeysRest], [Value|ValuesRest], NewAgent) :-
    put_dict(Key, Agent, Value, NewAgent1),
    modify_agent(NewAgent1, KeysRest, ValuesRest, NewAgent).


step_universe_turn(UniverseId, Action) :-
    get_current_agent_and_state(UniverseId, AgentId, StateId),
    state(StateId, Agents, _, _),
    Agent = Agents.get(AgentId),
    get_agent_action(UniverseId, AgentId, Action, Agent.name),
    step_universe_turn_with_action(UniverseId, Action).

step_universe_turn_with_action(UniverseId, [ActionHead|ActionArgs]) :-
    get_current_agent_and_state(UniverseId, AgentId, StateId),
    state(StateId, Agents, CurrentTurn, TurnOrder),
    history(StateId, UniverseId, Time, Turn),
    Agent = Agents.get(AgentId),
    can_perform(Agent.class, ActionHead),
    State = state(StateId, Agents, CurrentTurn, TurnOrder),
    (
        (ActionHead = rest ->
            % save mana (at most 100)
            RestedMana is Agent.mana + 1,
            (RestedMana > 100 -> NewMana = 100; NewMana = RestedMana),
            modify_agent(Agent, mana, NewMana, NewAgent),
            put_dict(AgentId, Agents, NewAgent, NewAgents),
            NewTurnOrder = TurnOrder);
        (ActionHead = move_right ->
            Xn is Agent.x + 1,
            \+tile_occupied(Xn, Agent.y, State),
            modify_agent(Agent, x, Xn, NewAgent),
            put_dict(AgentId, Agents, NewAgent, NewAgents),
            NewTurnOrder = TurnOrder
        );
        (ActionHead = move_up ->
            Yn is Agent.y + 1,
            \+tile_occupied(Agent.x, Yn, State),
            modify_agent(Agent, y, Yn, NewAgent),
            put_dict(AgentId, Agents, NewAgent, NewAgents),
            NewTurnOrder = TurnOrder
        );
        (ActionHead = move_left ->
            Xn is Agent.x - 1,
            \+tile_occupied(Xn, Agent.y, State),
            modify_agent(Agent, x, Xn, NewAgent),
            put_dict(AgentId, Agents, NewAgent, NewAgents),
            NewTurnOrder = TurnOrder
        );
        (ActionHead = move_down ->
            Yn is Agent.y - 1,
            \+tile_occupied(Agent.x, Yn, State),
            modify_agent(Agent, y, Yn, NewAgent),
            put_dict(AgentId, Agents, NewAgent, NewAgents),
            NewTurnOrder = TurnOrder
        );
        (ActionHead = portal ->
            % check whether global universe limit has been reached
            global_universe_id(GlobalUniverseId),
            universe_limit(UniverseLimit),
            GlobalUniverseId < UniverseLimit,
            % agent cannot time travel if there is only one agent in the universe
            length(TurnOrder, NumAgents),
            NumAgents > 1,
            [TargetUniverseId, TargetTime] = ActionArgs,
            % check whether target is now or in the past
            current_time(TargetUniverseId, TargetUniCurrentTime, _),
            TargetTime < TargetUniCurrentTime,
            % check whether there is enough mana
            (Agent.class = wizard -> TravelCost = 2; TravelCost = 5),
            Cost is abs(TargetTime - Time)*TravelCost + abs(TargetUniverseId - UniverseId)*TravelCost,
            Agent.mana >= Cost,
            % check whether the target location is occupied
            get_earliest_target_state(TargetUniverseId, TargetTime, TargetStateId),
            state(TargetStateId, TargetAgents, _, TargetTurnOrder),
            TargetState = state(TargetStateId, TargetAgents, _, TargetTurnOrder),
            \+tile_occupied(Agent.x, Agent.y, TargetState),
            % remove the agent from the current universe and turn order
            del_dict(AgentId, Agents, _, NewAgents),
            delete(TurnOrder, AgentId, NewTurnOrder),
            % remove cost from mana
            NewMana is Agent.mana - Cost,
            modify_agent(Agent, mana, NewMana, NewAgent),
            % put the agent into new parallel universe, and add it to the turn order (at the end)
            max_list(TargetTurnOrder, MaxTurnOrder),
            NewAgentId is MaxTurnOrder + 1,
            put_dict(NewAgentId, TargetAgents, NewAgent, NewTargetAgents),
            % increment global universe id
            NewGlobalUniverseId is GlobalUniverseId + 1,
            retract(global_universe_id(GlobalUniverseId)),
            assertz(global_universe_id(NewGlobalUniverseId)),
            % increment global state id
            global_state_id(TempGlobalStateId),
            TempNewGlobalStateId is TempGlobalStateId + 1,
            retract(global_state_id(TempGlobalStateId)),
            assertz(global_state_id(TempNewGlobalStateId)),
            % create new state and history
            assertz(state(TempNewGlobalStateId, NewTargetAgents, 1, [NewAgentId|TargetTurnOrder])),
            assertz(history(TempNewGlobalStateId, NewGlobalUniverseId, TargetTime, 1)),
            assertz(current_time(NewGlobalUniverseId, TargetTime, 1))
        );
        (ActionHead = portal_to_now ->
            % agent cannot time travel if there is only one agent in the universe
            length(TurnOrder, NumAgents),
            NumAgents > 1,
            [TargetUniverseId] = ActionArgs,
            % agent can only travel to now if it's the first turn in the target universe
            current_time(TargetUniverseId, TargetTime, 0),
            % agent cannot travel to current universe's now (would be a no-op)
            \+(TargetUniverseId = UniverseId),
            % check whether there is enough mana
            (Agent.class = wizard -> TravelCost = 2; TravelCost = 5),
            Cost is abs(TargetTime - Time)*TravelCost + abs(TargetUniverseId - UniverseId)*TravelCost,
            Agent.mana >= Cost,
            % check whether the target location is occupied
            get_latest_target_state(TargetUniverseId, TargetTime, TargetStateId),
            state(TargetStateId, TargetAgents, _, TargetTurnOrder),
            TargetState = state(TargetStateId, TargetAgents, _, TargetTurnOrder),
            \+tile_occupied(Agent.x, Agent.y, TargetState),
            % remove the agent from the current universe and turn order
            del_dict(AgentId, Agents, _, NewAgents),
            delete(TurnOrder, AgentId, NewTurnOrder),
            % remove cost from mana
            NewMana is Agent.mana - Cost,
            modify_agent(Agent, mana, NewMana, NewAgent),
            % put the agent into target universe, and add it to the turn order (at the end)
            max_list(TargetTurnOrder, MaxTurnOrder),
            NewAgentId is MaxTurnOrder + 1,
            put_dict(NewAgentId, TargetAgents, NewAgent, NewTargetAgents),
            % increment global state id
            global_state_id(TempGlobalStateId),
            TempNewGlobalStateId is TempGlobalStateId + 1,
            retract(global_state_id(TempGlobalStateId)),
            assertz(global_state_id(TempNewGlobalStateId)),
            % create new state and history
            assertz(state(TempNewGlobalStateId, NewTargetAgents, 1, [NewAgentId|TargetTurnOrder])),
            assertz(history(TempNewGlobalStateId, TargetUniverseId, TargetTime, 1)),
            assertz(current_time(TargetUniverseId, TargetTime, 1)),
            retract(current_time(TargetUniverseId, TargetTime, 0))
        );
        (ActionHead = melee_attack ->
            [TargetAgentId] = ActionArgs,
            TargetAgent = Agents.TargetAgentId,
            distance(Agent, TargetAgent, Distance),
            Distance =< 1,
            Damage is 20 - Agent.armor,
            TargetAgentHealth is TargetAgent.health - Damage,
            (TargetAgentHealth > 0 ->
                (modify_agent(TargetAgent, health, TargetAgentHealth, NewTargetAgent),
                 put_dict(TargetAgentId, Agents, NewTargetAgent, NewAgents),
                 NewTurnOrder = TurnOrder);
                (del_dict(TargetAgentId, Agents, _, NewAgents),
                 delete(TurnOrder, TargetAgentId, NewTurnOrder))
            )
        );
        (ActionHead = ranged_attack ->
            [TargetAgentId] = ActionArgs,
            TargetAgent = Agents.TargetAgentId,
            distance(Agent, TargetAgent, Distance),
            Distance =< 5,
            Damage is 15 - Distance - Agent.armor,
            TargetAgentHealth is TargetAgent.health - Damage,
            (TargetAgentHealth > 0 ->
                (modify_agent(TargetAgent, health, TargetAgentHealth, NewTargetAgent),
                 put_dict(TargetAgentId, Agents, NewTargetAgent, NewAgents),
                 NewTurnOrder = TurnOrder);
                (del_dict(TargetAgentId, Agents, _, NewAgents),
                 delete(TurnOrder, TargetAgentId, NewTurnOrder))
            )
        );
        (ActionHead = magic_missile ->
            [TargetAgentId] = ActionArgs,
            TargetAgent = Agents.TargetAgentId,
            distance(Agent, TargetAgent, Distance),
            Distance =< 10,
            Damage is 10 - Agent.agility,
            TargetAgentHealth is TargetAgent.health - Damage,
            (TargetAgentHealth > 0 ->
                (modify_agent(TargetAgent, health, TargetAgentHealth, NewTargetAgent),
                 put_dict(TargetAgentId, Agents, NewTargetAgent, NewAgents),
                 NewTurnOrder = TurnOrder);
                (del_dict(TargetAgentId, Agents, _, NewAgents),
                 delete(TurnOrder, TargetAgentId, NewTurnOrder))
            )
        )
    ),
    get_new_time_and_turn(UniverseId, NewTurnOrder, NewTime, NewTurn),
    global_state_id(GlobalStateId),
    NewGlobalStateId is GlobalStateId + 1,
    assertz(state(NewGlobalStateId, NewAgents, NewTurn, NewTurnOrder)),
    assertz(history(NewGlobalStateId, UniverseId, NewTime, NewTurn)),
    retract(global_state_id(GlobalStateId)),
    assertz(global_state_id(NewGlobalStateId)),
    retract(current_time(UniverseId, Time, Turn)),
    assertz(current_time(UniverseId, NewTime, NewTurn)).


step_universe_time(UniverseId, CurrentTime) :-
    step_universe_turn(UniverseId, _),
    current_time(UniverseId, NewTime, _),
    (NewTime = CurrentTime ->
        step_universe_time(UniverseId, CurrentTime);
        true
    ).


step_multiple_universes([], _).
step_multiple_universes([UniverseTimeTurn|Tail], MaxTime) :-
    UniverseTimeTurn = [UniverseId, Time, _],
    (Time = MaxTime ->
        (
            step_multiple_universes(Tail, MaxTime)
        );
        (
            step_universe_time(UniverseId, Time),
            step_multiple_universes(Tail, MaxTime)
        )
    ).

% The main loop.
main_loop(MaxTime) :-
    findall([Uid, Time, Turn], current_time(Uid, Time, Turn), Result),
    sort(Result, UniverseTimeTurns),
    step_multiple_universes(UniverseTimeTurns, MaxTime),
    findall(T, current_time(_, T, _), Times),
    min_list(Times, MinTime),
    (MinTime = MaxTime ->
        true;
        main_loop(MaxTime)
    ).
