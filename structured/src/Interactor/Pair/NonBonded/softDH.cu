#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"

#include "Interactor/Pair/PairInteractor.cuh"
#include "Interactor/Pair/NonBonded/NonBonded.cuh"
#include "Interactor/InteractorFactory.cuh"

#include "Interactor/BasicPotentials/LennardJones.cuh"
#include "Interactor/BasicParameters/Pair/LennardJones.cuh"
#include "Utils/ParameterHandler/PairParameterHandler.cuh"

namespace uammd{
namespace structured{
namespace Potentials{
namespace NonBonded{

    struct softDH_{

        using LennardJonesType      = typename BasicPotentials::LennardJones::Type1;

        using ParametersType        = typename BasicParameters::Pairs::LennardJones;
        using ParameterPairsHandler = typename structured::PairParameterHandler<ParametersType>;

        using ParametersPairsIterator = typename ParameterPairsHandler::PairIterator;

        struct ComputationalData{

            real4* pos;
	    real*  charge;
            Box box;

	    real cutOff;
	    real lambda;

	    real ELECOEF;
	    real debyeLength;
	    real dielectricConstant;
        };

        //Potential parameters
        struct StorageData{

            real cutOffFactor;
            real cutOff;
	    real lambda;

	    real ELECOEF;
	    real debyeLength;
	    real dielectricConstant;
        };

        //Computational data getter
        static __host__ ComputationalData getComputationalData(std::shared_ptr<GlobalData>    gd,
                                                               std::shared_ptr<ParticleGroup> pg,
                                                               const StorageData&  storage,
                                                               const Computables& comp,
                                                               const cudaStream_t& st){

            ComputationalData computational;

            std::shared_ptr<ParticleData> pd = pg->getParticleData();
	    computational.charge = pd->getCharge(access::location::gpu, access::mode::read).raw();

            computational.pos = pd->getPos(access::location::gpu, access::mode::read).raw();
            computational.box = gd->getEnsemble()->getBox();

	    computational.cutOff = storage.cutOff;
	    computational.lambda = storage.lambda;

	    computational.debyeLength = storage.debyeLength;
	    computational.ELECOEF = storage.ELECOEF;
	    computational.dielectricConstant = storage.dielectricConstant;

            return computational;
        }

        //Storage data reader

        static __host__ StorageData getStorageData(std::shared_ptr<GlobalData>    gd,
                                                   std::shared_ptr<ParticleGroup> pg,
                                                   DataEntry& data){

            StorageData storage;
            
            storage.cutOffFactor  = data.getParameter<real>("cutOffFactor");
            storage.lambda = data.getParameter<real>("lambda");
            
	    storage.debyeLength = data.getParameter<real>("debyeLength");;
	    storage.ELECOEF = gd->getUnits()->getElectricConversionFactor();
	    storage.dielectricConstant = data.getParameter<real>("dielectricConstant");;

            ///////////////////////////////////////////////////////////


            ///////////////////////////////////////////////////////////

            storage.cutOff = storage.debyeLength*storage.cutOffFactor;

            System::log<System::MESSAGE>("[softDH] cutOff: %f" ,storage.cutOff);

            return storage;
        }

        static inline __device__ real energy(int index_i, int index_j,
                                             const ComputationalData& computational){

            const real4 posi = computational.pos[index_i];
            const real4 posj = computational.pos[index_j];

            const real3 rij = computational.box.apply_pbc(make_real3(posj)-make_real3(posi));

            const real lambda = computational.lambda;

            const real r2 = dot(rij, rij);

            real e = real(0.0);

	    // DH
            real cutOff2 = computational.cutOff*computational.cutOff;
	    const real chgProduct = computational.charge[index_i]*computational.charge[index_j];
            if(r2<=cutOff2 and chgProduct != real(0.0)){
                real prefactorDH = computational.ELECOEF/computational.dielectricConstant*chgProduct;
		real lD = computational.debyeLength;
                real dist = sqrt(r2);
                e += prefactorDH*exp(-dist/lD)*lambda/(dist + (real(1.0)-lambda)*lD);
            }

            return e;

        }


        static inline __device__ real3 force(int index_i, int index_j,
                                             const ComputationalData& computational){

            const real4 posi = computational.pos[index_i];
            const real4 posj = computational.pos[index_j];

            const real3 rij = computational.box.apply_pbc(make_real3(posj)-make_real3(posi));
            
	    const real lambda = computational.lambda;

            const real r2 = dot(rij, rij);

            real3 f = make_real3(0.0);

	    // DH
            real cutOff2 = computational.cutOff*computational.cutOff;
	    const real chgProduct = computational.charge[index_i]*computational.charge[index_j];
            if(r2>0 and r2<=cutOff2 and chgProduct != real(0.0)){
                real prefactorDH = computational.ELECOEF/computational.dielectricConstant*chgProduct;
		real invLD = real(1.0)/computational.debyeLength;
                real dist = sqrt(r2);
                real invDistLambda = real(1.0)/(dist + (real(1.0) - lambda)*computational.debyeLength);
                real fmod = -prefactorDH*exp(-dist*invLD)*invDistLambda*lambda/dist*(invLD + invDistLambda);

                f += fmod*rij;
            }

            return f;
        }

      static inline __device__ tensor3 hessian(int index_i, int index_j,
					       const ComputationalData& computational){

	tensor3 H = tensor3(0.0);
	return H;
      }


    };

    using softDH = NonBondedHessian_<softDH_>;

}}}}

REGISTER_NONBONDED_INTERACTOR(
    NonBonded,softDH,
    uammd::structured::Interactor::PairInteractor<uammd::structured::Potentials::NonBonded::softDH>
)
