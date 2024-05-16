


graph = {'A': ['B', 'C'],
         'B': ['A', 'D', 'E'],
         'C': ['A', 'F'],
         'D': ['B', 'G'],
         'E': ['B', 'G'],
         'F': ['C', 'G'],
         'G': ['D', 'E', 'F']
}

class Agent():
    def __init__(self, graph, goal, start):
        self.graph = graph
        self.goal = goal
        self.frontier = [[start]]
        
    def next_path(self):
        pass
    
    def bfs(self):
        
        if len(self.frontier) == 0:
            return
            
        path = self.frontier.pop()
        
        current_node = path[-1]
        
        if current_node == self.goal:
            yield path
        
      
        for next_node in self.graph[current_node]:

            if next_node not in path:
                new_path = path.copy()
                new_path.append(next_node)

                self.frontier.append(new_path)
       
        yield from self.bfs()
           
            

        
        
        
    
def main():
    a = Agent(graph=graph, goal='C', start='A')
    
    for path in a.bfs():
        print('Path ',path)
        
if __name__ == '__main__':
    main()
    
  
            
        
            
        
        