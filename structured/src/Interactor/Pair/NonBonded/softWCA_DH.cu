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

    struct softWCA_DH_{

        using LennardJonesType      = typename BasicPotentials::LennardJones::Type1;

        using ParametersType        = typename BasicParameters::Pairs::LennardJones;
        using ParameterPairsHandler = typename structured::PairParameterHandler<ParametersType>;

        using ParametersPairsIterator = typename ParameterPairsHandler::PairIterator;

        struct ComputationalData{

            real4* pos;
	    real*  charge;
            Box box;

            ParametersPairsIterator paramPairIterator;

	    real cutOff;
	    real lambda;

	    real ELECOEF;
	    real debyeLength;
	    real dielectricConstant;

	    real alpha;
	    int n;
        };

        //Potential parameters
        struct StorageData{

            std::shared_ptr<ParameterPairsHandler> WCAParam;

            real cutOffFactor;
            real cutOff;
	    real lambda;

	    real ELECOEF;
	    real debyeLength;
	    real dielectricConstant;

	    real alpha;
	    int n;
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

            computational.paramPairIterator = storage.WCAParam->getPairIterator();
	    
	    computational.cutOff = storage.cutOff;
	    computational.lambda = storage.lambda;

	    computational.debyeLength = storage.debyeLength;
	    computational.ELECOEF = storage.ELECOEF;
	    computational.dielectricConstant = storage.dielectricConstant;

	    computational.alpha = storage.alpha;
	    computational.n = storage.n;

            return computational;
        }

        //Storage data reader

        static __host__ StorageData getStorageData(std::shared_ptr<GlobalData>    gd,
                                                   std::shared_ptr<ParticleGroup> pg,
                                                   DataEntry& data){

            StorageData storage;
            
	    storage.WCAParam = std::make_shared<ParameterPairsHandler>(gd,pg,data);

            storage.cutOffFactor  = data.getParameter<real>("cutOffFactor");
            storage.lambda = data.getParameter<real>("lambda");
            
	    storage.debyeLength = data.getParameter<real>("debyeLength");;
	    storage.ELECOEF = gd->getUnits()->getElectricConversionFactor();
	    storage.dielectricConstant = data.getParameter<real>("dielectricConstant");;

	    storage.alpha = data.getParameter<real>("alpha");;
	    storage.n = data.getParameter<real>("n");;

            ///////////////////////////////////////////////////////////


            ///////////////////////////////////////////////////////////

            auto pairsParam = storage.WCAParam->getPairParameters();

            real maxSigma = 0.0;
            for(auto p : pairsParam){
                maxSigma=std::max(maxSigma,p.second.sigma);
            }

            storage.cutOff = maxSigma*storage.cutOffFactor;

            System::log<System::MESSAGE>("[softWCA_DH] cutOff: %f" ,storage.cutOff);

            return storage;
        }

        static inline __device__ real energy(int index_i, int index_j,
                                             const ComputationalData& computational){

            const real4 posi = computational.pos[index_i];
            const real4 posj = computational.pos[index_j];

            const real3 rij = computational.box.apply_pbc(make_real3(posj)-make_real3(posi));

            const real lambda = computational.lambda;
            const real alpha = computational.alpha;
            const int n = computational.n;

            const real epsilon = computational.paramPairIterator(index_i,index_j).epsilon;
            const real sigma   = computational.paramPairIterator(index_i,index_j).sigma;

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

            // WCA
            const real Acomodo  = alpha*(real(1.0)-lambda)*(real(1.0)-lambda);
            const real minPos   = sigma*pow(real(1.0) - Acomodo, real(1.0)/real(6.0));
            const real minPos2  = minPos*minPos;            //position of minimum
            if (r2 < minPos2)
            {
		    real rnorm2 = r2/(sigma*sigma);
		    real fLambdaInv = real(1.0)/(Acomodo + rnorm2*rnorm2*rnorm2);
                    e += epsilon*pow(lambda, n)*(fLambdaInv*fLambdaInv - real(2.0)*fLambdaInv + real(1.0)); 
            }

            return e;

        }


        static inline __device__ real3 force(int index_i, int index_j,
                                             const ComputationalData& computational){

            const real4 posi = computational.pos[index_i];
            const real4 posj = computational.pos[index_j];

            const real3 rij = computational.box.apply_pbc(make_real3(posj)-make_real3(posi));
            
	    const real lambda = computational.lambda;
            const real alpha = computational.alpha;
            const int n = computational.n;

            const real epsilon = computational.paramPairIterator(index_i,index_j).epsilon;
            const real sigma   = computational.paramPairIterator(index_i,index_j).sigma;

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
                real fmod = -prefactorDH*exp(-dist*invLD)*lambda*invDistLambda/dist*(invLD + invDistLambda);

                f += fmod*rij;
            }

            // WCA
            const real Acomodo  = alpha*(real(1.0)-lambda)*(real(1.0)-lambda);
            const real minPos   = sigma*pow(real(1.0) - Acomodo, real(1.0)/real(6.0));
            const real minPos2  = minPos*minPos;            //position of minimum
            if (r2 < minPos2)
            {
		    real rnorm2 = r2/(sigma*sigma);
		    real rnorm6 = rnorm2*rnorm2*rnorm2;
		    real fLambda = Acomodo + rnorm6;
		    real fmod = -real(12.0)*epsilon*pow(lambda, n)/(r2*fLambda*fLambda*fLambda)*rnorm6*(real(1.0) - fLambda);
    
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

    using softWCA_DH = NonBondedHessian_<softWCA_DH_>;

}}}}

REGISTER_NONBONDED_INTERACTOR(
    NonBonded,softWCA_DH,
    uammd::structured::Interactor::PairInteractor<uammd::structured::Potentials::NonBonded::softWCA_DH>
)
