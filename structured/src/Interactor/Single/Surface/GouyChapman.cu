#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"

#include "Interactor/Single/SingleInteractor.cuh"
#include "Interactor/Single/Surface/Surface.cuh"
#include "Interactor/InteractorFactory.cuh"

#include "Interactor/BasicPotentials/Surface.cuh"

namespace uammd{
namespace structured{
namespace Potentials{
namespace Surface{

    struct GouyChapman_{

        struct ComputationalData{
            real4* pos;
            real* charge;
            real surfacePosition;
            real surfaceGamma;
            real debyeLength;
            real thermalEnergy;
        };

        //Potential parameters
        struct StorageData{
	    real surfacePosition;
            real surfaceGamma;
            real debyeLength;
            real thermalEnergy;
        };

        //Computational data getter
        static __host__ ComputationalData getComputationalData(std::shared_ptr<GlobalData>      gd,
                                                               std::shared_ptr<ParticleGroup> pg,
                                                               const StorageData&  storage,
                                                               const Computables& comp,
                                                               const cudaStream_t& st){

            ComputationalData computational;

            std::shared_ptr<ParticleData> pd = pg->getParticleData();
            computational.pos = pd->getPos(access::location::gpu, access::mode::read).raw();
            computational.charge = pd->getCharge(access::location::gpu, access::mode::read).raw();

            computational.surfacePosition = storage.surfacePosition;
            computational.surfaceGamma = storage.surfaceGamma;
            computational.debyeLength = storage.debyeLength;
            computational.thermalEnergy = storage.thermalEnergy;

            return computational;
        }

        //Storage data reader

        static __host__ StorageData getStorageData(std::shared_ptr<GlobalData>           gd,
                                                                 std::shared_ptr<ParticleGroup>        pg,
                                                                 DataEntry& data){

            StorageData storage;

            storage.surfacePosition = data.getParameter<real>("surfacePosition");
            storage.surfaceGamma = data.getParameter<real>("surfaceGamma");
            storage.debyeLength = data.getParameter<real>("debyeLength");
            storage.thermalEnergy = data.getParameter<real>("thermalEnergy");

            return storage;
        }


        static inline __device__  real3 force(const int& index_i,
                                              ComputationalData computational){

            real4 posi = computational.pos[index_i]; //position of particle i (i.e. coming from ''state'')
	    real qp = computational.charge[index_i];
	    real zs = computational.surfacePosition;
	    real gammas = computational.surfaceGamma;
	    real lD = computational.debyeLength;
	    real kBT = computational.thermalEnergy;

            real3 p = make_real3(posi);
	    real u = gammas*exp(-fabs(p.z - zs)/lD);
	    real forza = 4.0*qp*kBT/lD*u/(1.0-u*u);
	    if (p.z < zs)
		    forza *= -1.0;
	    
            real3 f = {0.0, 0.0, forza};

            return f;
        }

        static inline __device__  real energy(const int& index_i,
                                              ComputationalData computational){

            real4 posi = computational.pos[index_i]; //position of particle i (i.e. coming from ''state'')
	    real qp = computational.charge[index_i];
	    real zs = computational.surfacePosition;
	    real gammas = computational.surfaceGamma;
	    real lD = computational.debyeLength;
	    real kBT = computational.thermalEnergy;

            real3 p = make_real3(posi);
	    real u = gammas*exp(-fabs(p.z - zs)/lD);
            real e = -2.0*qp*kBT*log((1.0-u)/(1.0+u));

            return e;

        }

    };

    using GouyChapman = Surface_<GouyChapman_>;

}}}}

REGISTER_SINGLE_INTERACTOR(
    Surface,GouyChapman,
    uammd::structured::Interactor::SingleInteractor<uammd::structured::Potentials::Surface::GouyChapman>
)

