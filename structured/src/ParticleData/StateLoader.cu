#include "ParticleData/StateLoader.cuh"

namespace uammd{
namespace structured{

void stateLoader(ParticleData* pd,DataEntry& data){

    std::vector<std::string>   labels = data.getLabels();
    std::map<std::string,bool> isLabelLoaded;

    for(auto lbl:labels){
        isLabelLoaded[lbl] = false;
    }

    int N = pd->getNumParticles();

    //Id and position are compulsory

    std::vector<int> id2index(N);
    //Check id label is present
    if(std::find(labels.begin(), labels.end(), "id") == labels.end()){
        System::log<System::CRITICAL>("[StateLoader] Label 'id' not found in the state file.");
    } else {
        //Load id
        std::vector<int>  id = data.getData<int>("id");

        //Check there are N ids
        if(id.size() != N){
            System::log<System::CRITICAL>("[StateLoader] Number of ids in the state file does not match the number of particles.");
        }

        //Check id is unique
        std::sort(id.begin(), id.end());
        if(std::adjacent_find(id.begin(), id.end()) != id.end()){
            System::log<System::CRITICAL>("[StateLoader] Ids are not unique.");
        }

        //Check id is in the range [0,N-1]
        if(id[0] != 0 || id[N-1] != N-1){
            System::log<System::CRITICAL>("[StateLoader] Ids are not in the range [0,N-1].");
        }

        //Load id and create id2index
        System::log<System::MESSAGE>("[StateLoader] Loading property \"id\".");
        auto pId = pd->getId(access::location::cpu, access::mode::write);
        for(int i = 0; i < N; i++){
            pId[i] = id[i];
            id2index[id[i]] = i;
        }

        isLabelLoaded["id"] = true;
    }

    //Check position label is present
    if(std::find(labels.begin(), labels.end(), "position") == labels.end()){
        System::log<System::CRITICAL>("[StateLoader] Label 'position' not found in the state file.");
    } else {
        std::vector<real3> position = data.getData<real3>("position");

        //Load position
        System::log<System::MESSAGE>("[StateLoader] Loading property \"position\".");
        auto pPosition = pd->getPos(access::location::cpu, access::mode::write);
        for(int i = 0; i < N; i++){
            pPosition[i] = make_real4(position[id2index[i]],-1); //Note that the fourth component is the type, which is not loaded
        }

        isLabelLoaded["position"] = true;

    }


    if(std::find(labels.begin(), labels.end(), "velocity") != labels.end()){
        System::log<System::MESSAGE>("[StateLoader] Loading property \"velocity\".");
        std::vector<real3> velocity = data.getData<real3>("velocity");

        //Load velocity
        auto pVel = pd->getVel(access::location::cpu, access::mode::write);
        for(int i = 0; i < N; i++){
            pVel[i] = velocity[id2index[i]];
        }

        isLabelLoaded["velocity"] = true;
    }

    if(std::find(labels.begin(), labels.end(), "direction") != labels.end()){
        System::log<System::MESSAGE>("[StateLoader] Loading property \"direction\".");
        std::vector<real4> direction = data.getData<real4>("direction");

        //Load direction
        auto pDir = pd->getDir(access::location::cpu, access::mode::write);
        for(int i = 0; i < N; i++){
            pDir[i] = direction[id2index[i]];
        }

        isLabelLoaded["direction"] = true;
    }

    if(std::find(labels.begin(), labels.end(), "innerRadius") != labels.end()){
        System::log<System::MESSAGE>("[StateLoader] Loading property \"innerRadius\".");
        std::vector<real> innerRadius = data.getData<real>("innerRadius");

        //Load innerRadius
        auto pInnerRadius = pd->getInnerRadius(access::location::cpu, access::mode::write);
        for(int i = 0; i < N; i++){
            pInnerRadius[i] = innerRadius[id2index[i]];
        }

        isLabelLoaded["innerRadius"] = true;
    }

    if(std::find(labels.begin(), labels.end(), "magnetization") != labels.end()){
        System::log<System::MESSAGE>("[StateLoader] Loading property \"magnetization\".");
        std::vector<real4> magnetization = data.getData<real4>("magnetization");

        //Load magnetization
        auto pMagnetization = pd->getMagnetization(access::location::cpu, access::mode::write);
        for(int i = 0; i < N; i++){
            pMagnetization[i] = magnetization[id2index[i]];
        }

        isLabelLoaded["magnetization"] = true;
    }

    if(std::find(labels.begin(), labels.end(), "anisotropy") != labels.end()){
        System::log<System::MESSAGE>("[StateLoader] Loading property \"anisotropy\".");
        std::vector<real> anisotropy = data.getData<real>("anisotropy");

        //Load anisotropy
        auto pAnisotropy = pd->getAnisotropy(access::location::cpu, access::mode::write);
        for(int i = 0; i < N; i++){
            pAnisotropy[i] = anisotropy[id2index[i]];
        }

        isLabelLoaded["anisotropy"] = true;
    }

    if(std::find(labels.begin(), labels.end(), "mass") != labels.end()){
        System::log<System::MESSAGE>("[StateLoader] Loading property \"mass\".");
        std::vector<real> mass = data.getData<real>("mass");

        //Load mass
        auto pMass = pd->getMass(access::location::cpu, access::mode::write);
        for(int i = 0; i < N; i++){
            pMass[i] = mass[id2index[i]];
        }

        isLabelLoaded["mass"] = true;
    }

    if(std::find(labels.begin(), labels.end(), "polarizability") != labels.end()){
        System::log<System::MESSAGE>("[StateLoader] Loading property \"polarizability\".");
        std::vector<real> polarizability = data.getData<real>("polarizability");

        //Load polarizability
        auto pPolarizability = pd->getPolarizability(access::location::cpu, access::mode::write);
        for(int i = 0; i < N; i++){
            pPolarizability[i] = polarizability[id2index[i]];
        }

        isLabelLoaded["polarizability"] = true;
    }

    if(std::find(labels.begin(), labels.end(), "radius") != labels.end()){
        System::log<System::MESSAGE>("[StateLoader] Loading property \"radius\".");
        std::vector<real> radius = data.getData<real>("radius");

        //Load radius
        auto pRadius = pd->getRadius(access::location::cpu, access::mode::write);
        for(int i = 0; i < N; i++){
            pRadius[i] = radius[id2index[i]];
        }

        isLabelLoaded["radius"] = true;
    }

    if(std::find(labels.begin(), labels.end(), "charge") != labels.end()){
        System::log<System::MESSAGE>("[StateLoader] Loading property \"charge\".");
        std::vector<real> charge = data.getData<real>("charge");

        //Load charge
        auto pCharge = pd->getCharge(access::location::cpu, access::mode::write);
        for(int i = 0; i < N; i++){
            pCharge[i] = charge[id2index[i]];
        }

        isLabelLoaded["charge"] = true;
    }


    //Check all labels are loaded
    for(auto lbl:labels){
        if(!isLabelLoaded[lbl]){
            System::log<System::CRITICAL>("[StateLoader] \" %s \" is a not valid state property.",lbl.c_str());
        }
    }

}

void updateState(ParticleData* pd,DataEntry& data){
    System::log<System::DEBUG>("[StateLoader] Updating state.");

    std::vector<std::string> labels = data.getLabels();
    std::vector<std::string> toUpdate;


    if(pd->isVelAllocated()){
        toUpdate.push_back("velocity");
    }

    if(pd->isDirAllocated()){
        toUpdate.push_back("direction");
    }

    if(pd->isInnerRadiusAllocated()){
        toUpdate.push_back("innerRadius");
    }

    if(pd->isMagnetizationAllocated()){
        toUpdate.push_back("magnetization");
    }

    if(pd->isAnisotropyAllocated()){
        toUpdate.push_back("anisotropy");
    }

    if(pd->isMassAllocated()){
        toUpdate.push_back("mass");
    }

    if(pd->isPolarizabilityAllocated()){
        toUpdate.push_back("polarizability");
    }

    if(pd->isRadiusAllocated()){
        toUpdate.push_back("radius");
    }

    if(pd->isChargeAllocated()){
        toUpdate.push_back("charge");
    }


    //Remove all toUpdate labels that are present in labels
    for(auto lbl : labels){
        toUpdate.erase(std::remove(toUpdate.begin(), toUpdate.end(), lbl), toUpdate.end());
    }

    auto sortedIndex = pd->getIdOrderedIndices(access::location::cpu);

    //Add all toUpdate labels to labels
    labels.insert(labels.end(), toUpdate.begin(), toUpdate.end());
    //Set labels
    data.setLabels(labels);
    for(int i = 0; i < labels.size(); i++){
        std::string lbl = labels[i];

        if(lbl == "id"){
            auto pId = pd->getId(access::location::cpu, access::mode::read);
            for(int id=0;id<pd->getNumParticles();id++){
                int index = sortedIndex[id];
                data.setData(id, i, pId[index]);
            }
        }

        if(lbl == "position"){
            auto pPosition = pd->getPos(access::location::cpu, access::mode::read);
            for(int id=0;id<pd->getNumParticles();id++){
                int index = sortedIndex[id];
                data.setData(id, i, make_real3(pPosition[index]));
            }
        }


        if(lbl == "velocity"){
            auto pVel = pd->getVel(access::location::cpu, access::mode::read);
            for(int id=0;id<pd->getNumParticles();id++){
                int index = sortedIndex[id];
                data.setData(id, i, pVel[index]);
            }
        }

        if(lbl == "direction"){
            auto pDir = pd->getDir(access::location::cpu, access::mode::read);
            for(int id=0;id<pd->getNumParticles();id++){
                int index = sortedIndex[id];
                data.setData(id, i, pDir[index]);
            }
        }

        if(lbl == "innerRadius"){
            auto pInnerRadius = pd->getInnerRadius(access::location::cpu, access::mode::read);
            for(int id=0;id<pd->getNumParticles();id++){
                int index = sortedIndex[id];
                data.setData(id, i, pInnerRadius[index]);
            }
        }

        if(lbl == "magnetization"){
            auto pMagnetization = pd->getMagnetization(access::location::cpu, access::mode::read);
            for(int id=0;id<pd->getNumParticles();id++){
                int index = sortedIndex[id];
                data.setData(id, i, pMagnetization[index]);
            }
        }

        if(lbl == "anisotropy"){
            auto pAnisotropy = pd->getAnisotropy(access::location::cpu, access::mode::read);
            for(int id=0;id<pd->getNumParticles();id++){
                int index = sortedIndex[id];
                data.setData(id, i, pAnisotropy[index]);
            }
        }

        if(lbl == "mass"){
            auto pMass = pd->getMass(access::location::cpu, access::mode::read);
            for(int id=0;id<pd->getNumParticles();id++){
                int index = sortedIndex[id];
                data.setData(id, i, pMass[index]);
            }
        }

        if(lbl == "polarizability"){
            auto pPolarizability = pd->getPolarizability(access::location::cpu, access::mode::read);
            for(int id=0;id<pd->getNumParticles();id++){
                int index = sortedIndex[id];
                data.setData(id, i, pPolarizability[index]);
            }
        }

        if(lbl == "radius"){
            auto pRadius = pd->getRadius(access::location::cpu, access::mode::read);
            for(int id=0;id<pd->getNumParticles();id++){
                int index = sortedIndex[id];
                data.setData(id, i, pRadius[index]);
            }
        }

        if(lbl == "charge"){
            auto pCharge = pd->getCharge(access::location::cpu, access::mode::read);
            for(int id=0;id<pd->getNumParticles();id++){
                int index = sortedIndex[id];
                data.setData(id, i, pCharge[index]);
            }
        }


    }

}

}}
