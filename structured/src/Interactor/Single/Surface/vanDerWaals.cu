#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"

#include "Interactor/Single/SingleInteractor.cuh"
#include "Interactor/Single/Surface/Surface.cuh"
#include "Interactor/InteractorFactory.cuh"

#include "Interactor/BasicPotentials/Surface.cuh"
#include "Interactor/BasicParameters/Single/vanDerWaals.cuh"
#include "Utils/ParameterHandler/SingleParameterHandler.cuh"

namespace uammd{
namespace structured{
namespace Potentials{
namespace Surface{

    struct vanDerWaals_{

	using ParametersSingleType   = typename BasicParameters::Single::vanDerWaals;
	using ParameterSingleHandler = typename structured::SingleParameterHandler<ParametersSingleType>;
	using ParametersSingleIterator = typename ParameterSingleHandler::SingleIterator;

        struct ComputationalData{
            real4* pos;
	    real* radius;
            real surfacePosition;
	    real surfaceSqrtHamakerOverDensity;
	    ParametersSingleIterator paramSingleIterator;
        };

        //Potential parameters
        struct StorageData{
	    real surfacePosition;
	    real surfaceSqrtHamakerOverDensity;
	    std::shared_ptr<ParameterSingleHandler> info_vdW_WCA;
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
            computational.radius = pd->getRadius(access::location::gpu, access::mode::read).raw();

            computational.surfacePosition = storage.surfacePosition;
            computational.surfaceSqrtHamakerOverDensity = storage.surfaceSqrtHamakerOverDensity;
	    computational.paramSingleIterator = storage.info_vdW_WCA->getSingleIterator();

            return computational;
        }

        //Storage data reader

        static __host__ StorageData getStorageData(std::shared_ptr<GlobalData>           gd,
                                                                 std::shared_ptr<ParticleGroup>        pg,
                                                                 DataEntry& data){

            StorageData storage;

            storage.surfacePosition = data.getParameter<real>("surfacePosition");
            storage.surfaceSqrtHamakerOverDensity = data.getParameter<real>("surfaceSqrtHamakerOverDensity"); // sqrt(As/rhos)
	    storage.info_vdW_WCA = std::make_shared<ParameterSingleHandler>(gd, pg, data);

            return storage;
        }


        static inline __device__  real3 force(const int& index_i,
                                              ComputationalData computational){

	    // attractive (van der Waals)
            real4 posi = computational.pos[index_i]; //position of particle i (i.e. coming from ''state'')
	    real zs = computational.surfacePosition;
	    real surfaceSqrtHamakerOverDensity = computational.surfaceSqrtHamakerOverDensity;
	    real particleSqrtHamakerOverDensity = computational.paramSingleIterator(index_i).particleSqrtHamakerOverDensity;

            real3 p = make_real3(posi);
	    real dz = fabs(p.z - zs);
	    real dzq = dz*dz;
            real forza_vdW = -surfaceSqrtHamakerOverDensity*particleSqrtHamakerOverDensity/(real(2.0)*M_PI*dzq*dzq);
	    if (p.z < zs)
		    forza_vdW *= real(-1.0);
	    real3 f = make_real3(0.0);
	    f.z = forza_vdW;

	    // WCA
            real radiusi = computational.radius[index_i]; 
	    if (dz < radiusi)
	    {
	    	    real WCA_epsilon = computational.paramSingleIterator(index_i).WCA_epsilon;
		    real combo = real(1.0)*radiusi/dz;
		    real combo2 = combo*combo;
		    real combo6 = combo2*combo2*combo2;
		    real forza_WCA = real(12.0)*WCA_epsilon/(p.z - zs)*(combo6*combo6 - combo6);
		    f.z = f.z + forza_WCA;
	    }


            return f;
        }

        static inline __device__  real energy(const int& index_i,
                                              ComputationalData computational){

	    // attractive (van der Waals)
            real4 posi = computational.pos[index_i]; //position of particle i (i.e. coming from ''state'')
	    real zs = computational.surfacePosition;
	    real surfaceSqrtHamakerOverDensity = computational.surfaceSqrtHamakerOverDensity;
	    real particleSqrtHamakerOverDensity = computational.paramSingleIterator(index_i).particleSqrtHamakerOverDensity;

            real3 p = make_real3(posi);
	    real dz = fabs(p.z - zs);
            real e = -surfaceSqrtHamakerOverDensity*particleSqrtHamakerOverDensity/(real(6.0)*M_PI*dz*dz*dz);
	    
	    // WCA
            real radiusi = computational.radius[index_i]; 
	    if (dz < radiusi)
	    {
	    	    real WCA_epsilon = computational.paramSingleIterator(index_i).WCA_epsilon;
		    real combo = real(1.0)*radiusi/dz;
		    real combo2 = combo*combo;
		    real combo6 = combo2*combo2*combo2;
		    e += WCA_epsilon*(combo6*combo6 - real(2.0)*combo6) + WCA_epsilon;
	    }

            return e;

        }

    };

    using vanDerWaals = Surface_<vanDerWaals_>;

}}}}

REGISTER_SINGLE_INTERACTOR(
    Surface,vanDerWaals,
    uammd::structured::Interactor::SingleInteractor<uammd::structured::Potentials::Surface::vanDerWaals>
)

