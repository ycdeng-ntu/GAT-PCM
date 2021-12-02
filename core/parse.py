import xml.etree.ElementTree as ET
from core.agent import Agent


def parse(path, agent_type):
    agents = {}
    root = ET.parse(path).getroot()
    ele_agents = root.find('agents')
    for ele_agent in ele_agents.findall('agent'):
        id = ele_agent.get('name')
        neighbors = set()
        constraint_functions = {}
        agents[id] = agent_type(id, 0, neighbors, constraint_functions)
    domains = {}
    ele_domains = root.find('domains')
    for ele_domain in ele_domains.findall('domain'):
        id = ele_domain.get('name')
        nb_values = ele_domain.get('nbValues')
        domains[id] = int(nb_values)

    ele_variables = root.find('variables')
    for ele_variable in ele_variables.findall('variable'):
        agent_id = ele_variable.get('agent')
        domain_id = ele_variable.get('domain')
        agents[agent_id].domain = domains[domain_id]
    constraints = {}
    relations = {}
    ele_constraints = root.find('constraints')
    Agent.constraint_cnt = 0
    for ele_constraint in ele_constraints.findall('constraint'):
        Agent.constraint_cnt += 1
        id = ele_constraint.get('name')
        scope = ele_constraint.get('scope').split(' ')
        scope = ['A' + s[1: -2] for s in scope]
        reference = ele_constraint.get('reference')
        constraints[id] = scope
        relations[reference] = id
        for agent_id in scope:
            agents[agent_id].neighbors.update(scope)
            agents[agent_id].neighbors.remove(agent_id)

    ele_relations = root.find('relations')
    for ele_relation in ele_relations.findall('relation'):
        id = ele_relation.get('name')
        content = ele_relation.text.split('|')
        first_constraint = []
        for tpl in content:
            cost, values = tpl.split(':')
            cost = float(cost)
            values = [int(s) for s in values.split(' ')]
            while len(first_constraint) < values[0]:
                first_constraint.append([])
            row = first_constraint[values[0] - 1]
            while len(row) < values[1]:
                row.append(0)
            row[values[1] - 1] = cost
        second_constraint = []
        for i in range(len(first_constraint[0])):
            second_constraint.append([0] * len(first_constraint))
        for i in range(len(first_constraint)):
            for j in range(len(first_constraint[0])):
                second_constraint[j][i] = first_constraint[i][j]
        scope = constraints[relations[id]]
        agents[scope[0]].constraint_functions[scope[1]] = first_constraint
        agents[scope[1]].constraint_functions[scope[0]] = second_constraint
    return agents
