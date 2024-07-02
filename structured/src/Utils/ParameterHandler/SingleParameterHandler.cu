#include "Utils/ParameterHandler/SingleParameterHandler.cuh"

namespace uammd{
namespace structured{

    template<class SingleType>
    void SingleParameterHandler<class SingleType>::add(int t, int batchId, InputSingleParameters p){

        auto newp = SingleType::processSingleParameters(p);

        UAMMD_SET_2D_ROW_MAJOR(singleParameters_cpu, nBatches, nSingleTypes, batchId, t, newp);
        //UAMMD_SET_2D_COL_MAJOR(singleParameters_cpu, nBatches, nSingleTypes, batchId, t, newp);

        ////////////////////////////////////////

        std::pair<std::string,int> key = std::make_pair(p.name, batchId);

        //Check if the type is already in the map
        if(singleParameters.find(key) == singleParameters.end()){
            singleParameters[key] = newp;
        } else {
            System::log<System::CRITICAL>("[SingleParameterHandler] Trying to add a single parameter (%s) that already exists!", p.name.c_str());
        }

    }


    template<class SingleType>
    SingleParameterHandler<class SingleType>::SingleParameterHandler(std::shared_ptr<GlobalData>    gd,
                                                                     std::shared_ptr<ParticleGroup> pg,
                                                                     DataEntry& data):gd(gd), pg(pg),
                                                                                      pd(pg->getParticleData()),
                                                                                      types(gd->getTypes()){

        auto singleData = data.getDataMap();

        ///////////////////////////////////////////////////////////////////

        //Check if "batchId" is in labels
        std::vector<std::string> labels = data.getLabels();
        if(std::find(labels.begin(), labels.end(), "batchId") == labels.end()){
            System::log<System::MESSAGE>("[SingleParameterHandler] Parameters are not batched");
            nSingleTypes = types->getNumberOfTypes();

            isBatched = false;
            nBatches  = 1;

        } else {
            System::log<System::MESSAGE>("[SingleParameterHandler] Parameters are batched");

            if(!Batching::checkDataConsistency(data)){
                System::log<System::CRITICAL>("[SingleParameterHandler] Parameters are not consistent");
            }

            if(!Batching::checkParticleGroupDataConsistency(pg, data)){
                System::log<System::CRITICAL>("[SingleParameterHandler] Parameters are not consistent with particle group");
            }

            nSingleTypes = types->getNumberOfTypes();

            isBatched = true;
            std::vector<int> batchIds = data.getData<int>("batchId");
            //Count number of different batch ids. Convert to set to remove duplicates
            nBatches = std::set<int>(batchIds.begin(), batchIds.end()).size();
        }

        System::log<System::MESSAGE>("[SingleParameterHandler] Number of single types: %d", nSingleTypes);
        System::log<System::MESSAGE>("[SingleParameterHandler] Number of batches: %d", nBatches);

        //Resize
        singleParameters_cpu.resize(nSingleTypes*nBatches);

        ///////////////////////////////////////////////////////////////////

        InputSingleParameters singleParamBuffer;
        for(auto singleInfo : singleData){

            singleParamBuffer = SingleType::readSingleParameters(singleInfo);
            std::string name = singleParamBuffer.name;

            int t = types->getTypeId(name);

            if(isBatched){
                int batchId = singleInfo["batchId"];

                this->add(t,batchId,singleParamBuffer);
                System::log<System::DEBUG>("[SingleParameterHandler] Single %s (type id: %i, batch id: %i) added",
                        name.c_str(),
                        t,batchId);
            } else {
                this->add(t,0,singleParamBuffer);
                System::log<System::DEBUG>("[SingleParameterHandler] Single %s (type id: %i) added.",
                        name.c_str(),
                        t);
            }
        }

        ///////////////////////////////////////////////////////////////////

        singleParameters_gpu = singleParameters_cpu;

    }

    template<class SingleType>
    SingleParameterHandler<class SingleType>::SingleIterator SingleParameterHandler<class SingleType>::getSingleIterator(){

        auto pos   = pd->getPos(access::location::gpu, access::mode::read);
        auto batchId = pd->getBatchId(access::location::gpu, access::mode::read);

        auto sp = thrust::raw_pointer_cast(singleParameters_gpu.data());

        return SingleIterator(pos.raw(), batchId.raw(), sp, nSingleTypes, nBatches);
    }

}}

