#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"

#include "Interactor/Bonds/BondsInteractor.cuh"
#include "Interactor/Bonds/Bond1/Bond1.cuh"
#include "Interactor/InteractorFactory.cuh"

#include "Interactor/BasicPotentials/Harmonic.cuh"

namespace uammd{
namespace structured{
namespace Potentials{
namespace Bond1{

    struct TangentialActiveForce_{

        struct ComputationalData{
            real4* pos;
	    const int* id2index;
	    Box box;
        };

        //Potential parameters

        struct StorageData{
	};

	struct BondParameters{
            real F0;
	    int i1;
	    int i2;
        };

        //Computational data getter

        static __host__ ComputationalData getComputationalData(std::shared_ptr<GlobalData>    gd,
                                                               std::shared_ptr<ParticleGroup> pg,
                                                               const StorageData&  storage,
                                                               const Computables& computables,
                                                               const cudaStream_t& st){

            ComputationalData computational;

            std::shared_ptr<ParticleData> pd = pg->getParticleData();

            computational.pos      = pd->getPos(access::location::gpu, access::mode::read).raw();
	    computational.id2index = pd->getIdOrderedIndices(access::location::gpu);
	    computational.box      = gd->getEnsemble()->getBox();

            return computational;
        }

        //Storage data reader

        static __host__ StorageData getStorageData(std::shared_ptr<GlobalData>    gd,
                                                   std::shared_ptr<ParticleGroup> pg,
                                                   DataEntry& data){

            StorageData storage;
            return storage;
        }

        //Bond parameters reader

        template<typename T>
        static __host__ BondParameters processBondParameters(std::shared_ptr<GlobalData> gd,
                                                             std::map<std::string,T>& bondParametersMap){

            BondParameters param;

            param.F0   = bondParametersMap.at("F0");
            param.i1   = bondParametersMap.at("id_j");
            param.i2   = bondParametersMap.at("id_k");

            return param;
        }

        //Energy and force definition

        static inline __device__ real3 force(int index_i,
                                             int currentParticleIndex,
                                             const ComputationalData &computational,
                                             const BondParameters &bondParam){

	    Box box           = computational.box;
	    const int index_j = computational.id2index[bondParam.i1];
	    const int index_k = computational.id2index[bondParam.i2];
            const real3 posi  = make_real3(computational.pos[index_i]);
            const real3 posj  = make_real3(computational.pos[index_j]);
            const real3 posk  = make_real3(computational.pos[index_k]);
            const real  F0    = real(bondParam.F0);

	    real3 vec;
	    real3 vecq;
	    vec = box.apply_pbc(posk - posj);
	    vec = normalize(vec);
	    const real3 f = make_real3(F0*vec);

	    //real3 f = make_real3(0);
	    //f.x = 2.34;
            return f;
        }

        static inline __device__ real energy(int index_i,
                                             int currentParticleIndex,
                                             const ComputationalData &computational,
                                             const BondParameters &bondParam){

            return 0.0;
        }

      static inline __device__ tensor3 hessian(int index_i,
					       int currentParticleIndex,
					       const ComputationalData &computational,
					       const BondParameters &bondParam){

	tensor3 H = tensor3();
	return H;
        }

    };

    using TangentialActiveForce = Bond1Hessian_<TangentialActiveForce_>;

}}}}

REGISTER_BOND_INTERACTOR(
    Bond1,TangentialActiveForce,
    uammd::structured::Interactor::BondsInteractor<uammd::structured::Potentials::Bond1::TangentialActiveForce>
)
