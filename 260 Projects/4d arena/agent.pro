get_agent_action(UniverseId, AgentId, Action, _) :- random_policy(UniverseId, AgentId, Action).

get_agent_action(UniverseId, AgentId, Action, 'Arthur') :-
    random_policy(UniverseId, AgentId, Action).
    % get_current_agent_and_state(UniverseId, AgentId, StateId),
    % basic_action_policy(StateId, AgentId, Action).

get_agent_action(UniverseId, AgentId, Action, 'Merlin') :-
    random_policy(UniverseId, AgentId, Action).
    % get_current_agent_and_state(UniverseId, AgentId, StateId),
    % basic_action_policy(StateId, AgentId, Action).

get_agent_action(UniverseId, AgentId, Action, 'Mordred') :-
    random_policy(UniverseId, AgentId, Action).
    % get_current_agent_and_state(UniverseId, AgentId, StateId),
    % basic_action_policy(StateId, AgentId, Action).

random_policy(UniverseId, AgentId, Action) :-
    get_current_agent_and_state(UniverseId, AgentId, StateId),
    state(StateId, Agents, _, _),
    Agent = Agents.get(AgentId),
    findall(A, can_perform(Agent.class, A), ActionList),
    repeat,
    random_member(ActionProposal, ActionList),
    % pick a random time and universe if action is portal
    % pick a random agent id for attack if action is one of
    % attack .
    (
        (ActionProposal = portal,
         global_universe_id(MaxUniverseId),
         findall(Uid, between(0, MaxUniverseId, Uid), UniverseIds),
         random_member(UniverseId, UniverseIds),
         current_time(UniverseId, MaxTime, _),
         findall(T, between(0, MaxTime, T), Times),
         random_member(Time, Times),
         Action = [portal, UniverseId, Time]);
        (ActionProposal = portal_to_now,
         global_universe_id(MaxUniverseId),
         findall(Uid, between(0, MaxUniverseId, Uid), UniverseIds),
         random_member(UniverseId, UniverseIds),
         Action = [portal_to_now, UniverseId]);
        (member(ActionProposal, [melee_attack, magic_missile, ranged_attack]),
         dict_keys(Agents, AgentIds),
         random_member(TargetAgentId, AgentIds),
         \+ (TargetAgentId = AgentId),
         Action = [ActionProposal, TargetAgentId]);
        (Action = [ActionProposal])
    ).
