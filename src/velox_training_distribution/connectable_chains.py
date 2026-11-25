"""Find native paths with two witnessed terminal ports and distinct native subjects."""
import itertools
from collections import defaultdict,deque


class WitnessSearch:
    def __init__(self,db):
        self.graphs=defaultdict(lambda:defaultdict(set))
        self.ports=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
        self.potential=defaultdict(set)
        self.verified=set()
        for w,u,z in db.execute("""select distinct w,uid,z from sites
                union select w,src_uid,dst_uid from forks""").fetchall():
            graph=self.graphs[w]
            graph[u].add(z)
            graph[z].add(u)
        for h,t,u,w,z,du in db.execute('select * from ends').fetchall():
            self.ports[w][u][z][h,t]=du
        for w,graph in self.graphs.items():
            seen=set()
            for start in self.ports[w]:
                if start in seen:continue
                comp={start};q=[start]
                while q:
                    for v in graph[q.pop()]:
                        if v not in comp:comp.add(v);q.append(v)
                seen.update(comp)
                types=defaultdict(set)
                for u in comp.intersection(self.ports[w]):
                    for port in self.ports[w][u].values():
                        for h,t in port:types[h].add(t)
                for h,ts in types.items():
                    self.potential[h].update(itertools.combinations_with_replacement(sorted(ts),2))

    @staticmethod
    def middle(graph,x,y,exclude,all_paths):
        if x==y:
            yield (x,)
            return
        parents={x:[]};dist={x:0};q=deque([x])
        while q:
            u=q.popleft()
            if y in dist and dist[u]>=dist[y]:break
            for v in graph[u]:
                if v in exclude:continue
                if v not in dist:
                    parents[v]=[u];dist[v]=dist[u]+1;q.append(v)
                elif all_paths and dist[v]==dist[u]+1:parents[v].append(u)
        stack=[(y,(y,))] if y in dist else []
        while stack:
            u,path=stack.pop()
            if u==x:yield path
            else:stack.extend((v,(v,)+path) for v in parents[u])

    def paths(self,chosen,direct,validate=False):
        want={(h,a,b) for h,ts in chosen.items() for a,b in itertools.combinations_with_replacement(sorted(ts),2)}
        if validate:
            want={k for k in want if k[1]!=k[2] and k[1:] not in direct[k[0]]}-self.verified
        for w,by_uid in self.ports.items():
            if not want:break
            ports={}
            for u,neighbors in by_uid.items():
                for z,types in neighbors.items():
                    items={(h,t):du for (h,t),du in types.items() if t in chosen[h]}
                    if items:ports.setdefault(u,{})[z]=items
            for u,v in itertools.combinations(sorted(ports),2):
                for x,aa in ports[u].items():
                    if x==v:continue
                    for y,bb in ports[v].items():
                        if y==u:continue
                        options=[]
                        for (h,a),da in aa.items():
                            for (hh,b),db in bb.items():
                                if h!=hh:continue
                                key=(h,min(a,b),max(a,b))
                                if key not in want:continue
                                options.append((key,a>b,da,db))
                        if not options:continue
                        for center in self.middle(self.graphs[w],x,y,{u,v},not validate):
                            yield w,(u,)+center+(v,),options
                            if validate:
                                want.difference_update(k for k,*_ in options)
                                break

    def validate(self,chosen,direct):
        for _,_,opts in self.paths(chosen,direct,True):
            self.verified.update(k for k,*_ in opts)
        return [(h,a,b) for h,ts in chosen.items() for a,b in itertools.combinations(sorted(ts),2)
                if (a,b) not in direct[h] and (h,a,b) not in self.verified]
