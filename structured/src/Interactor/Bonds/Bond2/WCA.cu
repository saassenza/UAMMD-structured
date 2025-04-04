#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"

#include "Interactor/Bonds/BondsInteractor.cuh"
#include "Interactor/Bonds/Bond2/Bond2.cuh"
#include "Interactor/InteractorFactory.cuh"

namespace uammd{
namespace structured{
namespace Potentials{
namespace Bond2{

    struct WCA_{

        struct ComputationalData{
            real4* pos;
            Box box;
	    real epsilon;
            real sigma;
	    real cutOff;
        };

        //Potential parameters

        struct StorageData{
	    real epsilon;
            real sigma;
        };

        struct BondParameters{

        };

        //Computational data getter

        static __host__ ComputationalData getComputationalData(std::shared_ptr<GlobalData>    gd,
                                                               std::shared_ptr<ParticleGroup> pg,
                                                               const StorageData&  storage,
                                                               const Computables& computables,
                                                               const cudaStream_t& st){

            ComputationalData computational;

            std::shared_ptr<ParticleData> pd = pg->getParticleData();

            computational.pos     = pd->getPos(access::location::gpu, access::mode::read).raw();
            computational.box     = gd->getEnsemble()->getBox();

	    computational.epsilon = storage.epsilon;
            computational.sigma   = storage.sigma;
	    computational.cutOff  = storage.sigma*1.12246204831; //2^(1./6)

            return computational;
        }

        //Storage data reader

        static __host__ StorageData getStorageData(std::shared_ptr<GlobalData>    gd,
                                                   std::shared_ptr<ParticleGroup> pg,
                                                   DataEntry& data){

            StorageData storage;
	    storage.epsilon = data.getParameter<real>("epsilon");
            storage.sigma   = data.getParameter<real>("sigma");
            return storage;
        }

        //Bond parameters reader

        template<typename T>
        static __host__ BondParameters processBondParameters(std::shared_ptr<GlobalData> gd,
                                                             std::map<std::string,T>& bondParametersMap){

            BondParameters param;
            return param;
        }

        //Energy and force definition

        static inline __device__ real energy(int index_i, int index_j,
                                             int currentParticleIndex,
                                             const ComputationalData &computational,
                                             const BondParameters   &bondParam){

            real3 posi = make_real3(computational.pos[index_i]);
            real3 posj = make_real3(computational.pos[index_j]);

            real3 rij = computational.box.apply_pbc(posj-posi);

            real  r2 = dot(rij, rij);

            real e=real(0.0);
	    real sr2;
 	    real sr6;	    
            if(r2<(computational.cutOff*computational.cutOff)){
		sr2 = real(1.0)*computational.sigma*computational.sigma/r2;
		sr6 = sr2*sr2*sr2;
                e   = real(4.0)*computational.epsilon*(sr6*sr6 - sr6) + computational.epsilon;
            }

            return e;
        }

        static inline __device__ real3 force(int index_i, int index_j,
                                             int currentParticleIndex,
                                             const ComputationalData &computational,
                                             const BondParameters   &bondParam){

            real3 posi = make_real3(computational.pos[index_i]);
            real3 posj = make_real3(computational.pos[index_j]);

            real3 rij = computational.box.apply_pbc(posj-posi);

            real  r2 = dot(rij, rij);

            real3 f=make_real3(0.0);
	    real sr2;
            real sr6;
	    real prefactorWCA;
            if(r2<(computational.cutOff*computational.cutOff)){
                sr2 = real(1.0)*computational.sigma*computational.sigma/r2;
                sr6 = sr2*sr2*sr2;
		prefactorWCA = real(24.0)*computational.epsilon/r2*(sr6 - 2*sr6*sr6);
		f            = prefactorWCA*rij;
                if        (currentParticleIndex == index_i){
                } else if (currentParticleIndex == index_j){
                    f=-f;
                }
            }

            return f;

        }


    };

    using WCA = Bond2_<WCA_>;

}}}}

REGISTER_BOND_INTERACTOR(
    Bond2,WCA,
    uammd::structured::Interactor::BondsInteractor<uammd::structured::Potentials::Bond2::WCA>
)
