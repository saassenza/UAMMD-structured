#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"
#include "ParticleGroup/ParticleGroupUtils.cuh"

#include "DataStructures/VerletConditionalListSet/VerletConditionalListSet.cuh"
#include "DataStructures/VerletConditionalListSet/VerletConditionalListSetFactory.cuh"
#include "DataStructures/VerletConditionalListSet/Condition/Condition.cuh"

namespace uammd{
namespace structured{
namespace conditions{

    class NonExclIdGroup1Intra_NonExclIdGroup2Intra_NonExclInter_NonExclNoGroup : public excludedConditionBase {

            std::vector<int> idGroup1_h;
            std::vector<int> idGroup2_h;

            thrust::device_vector<int> idGroup1_d;
            thrust::device_vector<int> idGroup2_d;

        public:

            /////////////////////////////

            static const int condNum = 6;
            enum condIndex {ID_GROUP1_INTRA=0,  //only internal to group1
                            ID_GROUP2_INTRA=1,  //only internal to group2
                            INTRA=2,            //only internal to groups (each group to itself)
                            INTERGROUPS=3,      //inter groups (has priority over models)
                            INTERMODELS=4,      //inter models 
                            NOGROUP=5};

            /////////////////////////////

            NonExclIdGroup1Intra_NonExclIdGroup2Intra_NonExclInter_NonExclNoGroup(std::shared_ptr<GlobalData>            gd,
                                                                                      std::shared_ptr<ExtendedParticleData>  pd,
                                                                                      DataEntry& data):excludedConditionBase(gd,pd,data){

                std::vector<int> idGroup1 = data.getParameter<std::vector<int>>("idGroup1");
                std::vector<int> idGroup2 = data.getParameter<std::vector<int>>("idGroup2");

                auto typeParamHandler = gd->getTypes();

                for(auto& id : idGroup1){
                    idGroup1_h.push_back(id);
                }

                for(auto& id : idGroup2){
                    idGroup2_h.push_back(id);
                }

                // Sort idGroup1 and idGroup2
                std::sort(idGroup1_h.begin(),idGroup1_h.end());
                std::sort(idGroup2_h.begin(),idGroup2_h.end());

                idGroup1_d = idGroup1_h;
                idGroup2_d = idGroup2_h;

                System::log<System::MESSAGE>("[Condition] Condition "
                                             "\"nonExclTypeGroup1Intra_nonExclTypeGroup2Intra_nonExcInter_nonExclNoGroup\" "
                                             "initialized.");

            }

            ////////////////////////////////////////////

            int getConditionIndexOf(std::string& conditionName){

                if     (conditionName=="idGroup1Intra"){return ID_GROUP1_INTRA;}
                else if(conditionName=="idGroup2Intra"){return ID_GROUP2_INTRA;}
                else if(conditionName=="intra")        {return INTRA;}
                else if(conditionName=="interGroups")  {return INTERGROUPS;}
                else if(conditionName=="interModels")  {return INTERMODELS;}
                else if(conditionName=="noGroup")      {return NOGROUP;}
                else
                {
                    System::log<System::CRITICAL>("[Condition] Requested a condition"
                                                  " that is not present, %s",
                                                  conditionName.c_str());
                    return -1;
                }

                return -1;

            }

            ////////////////////////////////////////////

            struct conditionChecker{

                int* id;
                int* modelId;

                real4*  pos;

                int* idGroup1;
                int* idGroup2;

                int idGroup1Size;
                int idGroup2Size;

                particleExclusionList prtExclList;
                int maxExclusions;

                __device__ int binarySearch(const int* data, int size, int value) {
                    int left = 0;
                    int right = size - 1;

                    while (left <= right) {
                        int middle = left + (right - left) / 2;

                        if (data[middle] == value) {
                            return middle; // or return a specific flag indicating found
                        }
                        if (data[middle] < value) {
                            left = middle + 1;
                        }
                        else {
                            right = middle - 1;
                        }
                    }

                    return -1; // or return a specific flag indicating not found
                }

                inline __device__ int getGroup(int id){

                    if     (binarySearch(idGroup1,idGroup1Size,id)!=-1){
                        return 1;
                    }
                    else if(binarySearch(idGroup2,idGroup2Size,id)!=-1){
                        return 2;
                    }
                    else{
                        return 0;
                    }

                    return 0;

                }

                inline __device__ void set(const int& i,const int& offsetBufferIndex, const char* sharedBuffer){
                    prtExclList.set(id[i],(int*)sharedBuffer+offsetBufferIndex*maxExclusions);
                }

                inline __device__ void check(const int& i,const int& j,bool cond[condNum]){

                    for(int c=0;c<condNum;c++){
                        cond[c] = false;
                    }

                    if(!prtExclList.isPartExcluded(id[j])){

                        int modelId_i = modelId[i];
                        int modelId_j = modelId[j];

                        int id_i = int(id[i]);
                        int id_j = int(id[j]);

                        int group_i = getGroup(id_i);
                        int group_j = getGroup(id_j);

                        bool noGroup = (group_i==0 || group_j==0);

                        if(noGroup){
                            cond[NOGROUP] = true;
                        }else{
                            if(group_i!=group_j){
                                cond[INTERGROUPS] = true;
                            }
                            else{
                            	if (modelId_i != modelId_j){
                                    cond[INTERMODELS] = true;
                                } else{
                                    if(group_i==1){
                                        cond[ID_GROUP1_INTRA] = true;
					cond[INTRA] = true;
                                    }
                                    else{
                                        cond[ID_GROUP2_INTRA] = true;
					cond[INTRA] = true;
                                    }
                                }
                            }
                        }
                    }
                }

                conditionChecker(int* id, int* modelId,
                                 real4* pos,
                                 int* idGroup1,int* idGroup2,
                                 int idGroup1Size,int idGroup2Size,
                                 particleExclusionList prtExclList,
                                 int maxExclusions):
                                 id(id),modelId(modelId),
                                 pos(pos),
                                 idGroup1(idGroup1),idGroup2(idGroup2),
                                 idGroup1Size(idGroup1Size),idGroup2Size(idGroup2Size),
                                 prtExclList(prtExclList),
                                 maxExclusions(maxExclusions){}

            };


            conditionChecker getConditionChecker(){

                int*      id = pd->getId(access::location::gpu,access::mode::read).raw();
                int* modelId = pd->getModelId(access::location::gpu,access::mode::read).raw();

                real4* pos = pd->getPos(access::location::gpu,access::mode::read).raw();

                int* idGroup1 = thrust::raw_pointer_cast(idGroup1_d.data());
                int* idGroup2 = thrust::raw_pointer_cast(idGroup2_d.data());

                return conditionChecker(id,modelId,
                                        pos,
                                        idGroup1,idGroup2,
                                        idGroup1_d.size(),idGroup2_d.size(),
                                        exclusionList->getParticleExclusionList(),
                                        exclusionList->getMaxExclusions());
            }
    };

}}}

REGISTER_VERLET_CONDITIONAL_LIST_SET(
    NonExclIdGroup1Intra_NonExclIdGroup2Intra_NonExclInter_NonExclNoGroup,
    uammd::structured::VerletConditionalListSet<uammd::structured::conditions::NonExclIdGroup1Intra_NonExclIdGroup2Intra_NonExclInter_NonExclNoGroup>
)
