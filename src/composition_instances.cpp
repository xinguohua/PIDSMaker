// Stream strictly ordered native paths per window. Store type counts and event memberships.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
using Key=uint64_t;
constexpr int SHIFT=21;
struct Event { int row,u,v,group,action,complete; };
struct Counts { uint64_t train=0,test=0; };
struct Result { unsigned mask=0; Key witness=0; };
int main(int argc,char**argv){
    if(argc!=2) return 2;
    std::string dir=argv[1];
    std::ifstream in(dir+"/events.tsv");
    std::ofstream membership(dir+"/event_groups.tsv"), types(dir+"/types.bin",std::ios::binary);
    std::unordered_map<Key,Counts> counts;
    std::unordered_map<int,uint64_t> actions;
    int w,test,size;
    while(in>>w>>test>>size){
        std::vector<Event> es(size);
        std::unordered_map<int,std::vector<int>> outgoing;
        for(int i=0;i<size;i++){
            auto &e=es[i];
            in>>e.row>>e.u>>e.v>>e.group>>e.action>>e.complete;
            assert(in && e.action>0 && e.action<(1<<SHIFT));
            outgoing[e.u].push_back(i);
            if(!test) actions[e.action]++;
        }
        std::vector<std::array<Result,2>> flags(size);
        auto record=[&](int i,int j,int k){
            Key key=(Key(es[i].action)<<SHIFT)|es[j].action;
            if(k>=0) key=(key<<SHIFT)|es[k].action;
            Counts &c=counts[key];
            if(!test){++c.train;return;}
            ++c.test;
            bool complete=es[i].complete && es[j].complete && (k<0 || es[k].complete);
            bool known=actions.count(es[i].action) && actions.count(es[j].action) && (k<0 || actions.count(es[k].action));
            unsigned bit=!complete?8:!known?4:!c.train?2:1;
            if(bit==1) bit |= c.train<=10?16:c.train<=100?32:64;
            for(int pos:{i,j,k}) if(pos>=0){
                Result &r=flags[pos][k>=0];
                r.mask|=bit;
                // First path in fixed traversal order supplies one outcome-blind control witness.
                if(!r.witness) r.witness=key;
            }
        };
        auto later=[&](const std::vector<int>& ids,int group){
            return std::upper_bound(ids.begin(),ids.end(),group,[&](int g,int id){return g<es[id].group;});
        };
        for(int i=0;i<size;i++){
            auto mid=outgoing.find(es[i].v);
            if(mid==outgoing.end()) continue;
            for(auto j=later(mid->second,es[i].group);j!=mid->second.end();++j){
                record(i,*j,-1);
                auto tail=outgoing.find(es[*j].v);
                if(tail==outgoing.end()) continue;
                for(auto k=later(tail->second,es[*j].group);k!=tail->second.end();++k) record(i,*j,*k);
            }
        }
        if(test) for(int i=0;i<size;i++)
            membership<<w<<' '<<es[i].row<<' '<<flags[i][0].mask<<' '<<flags[i][1].mask<<' '<<flags[i][0].witness<<' '<<flags[i][1].witness<<'\n';
    }
    if(!in.eof()) return 3;
    for(auto &item:counts){
        std::array<Key,3> data={item.first,item.second.train,item.second.test};
        types.write(reinterpret_cast<const char*>(data.data()),sizeof(data));
    }
    return !types || !membership;
}
