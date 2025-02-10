#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"

#include "Interactor/Pair/PairInteractor.cuh"
#include "Interactor/Pair/NonBonded/NonBonded.cuh"
#include "Interactor/InteractorFactory.cuh"

#include "Interactor/BasicPotentials/DebyeHuckel.cuh"
#include "Interactor/BasicPotentials/LennardJones.cuh"
#include "Interactor/BasicParameters/Pair/WCA_DH.cuh"
#include "Utils/ParameterHandler/PairParameterHandler.cuh"

namespace uammd{
namespace structured{
namespace Potentials{
namespace NonBonded{

    struct softWCA_DH_{

	using ParametersType        = typename BasicParameters::Pairs::WCA_DH;
	using ParameterPairsHandler = typename structured::PairParameterHandler<ParametersType>;
        using ParametersPairsIterator = typename ParameterPairsHandler::PairIterator;

        //Computational data
        struct ComputationalData{

            real4* pos;

            Box    box;

            real ELECOEF;

            real dielectricConstant;
            real debyeLength;
	    real lambda;
            real alpha;
            int  n;

	    ParametersPairsIterator paramPairIterator;

            real cutOffFactor;
            real cutOff;
        };

        //Potential parameters
        struct StorageData{

	    std::shared_ptr<ParameterPairsHandler> Param;
            real ELECOEF;

            real dielectricConstant;
            real debyeLength;
            real alpha;
            int  n;

            real cutOffFactor;
            real cutOff;
        };

        static __host__ ComputationalData getComputationalData(std::shared_ptr<GlobalData>    gd,
                                                               std::shared_ptr<ParticleGroup> pg,
                                                               const StorageData&  storage,
                                                               const Computables& comp,
                                                               const cudaStream_t& st){

            ComputationalData computational;

            std::shared_ptr<ParticleData> pd = pg->getParticleData();

            computational.pos    = pd->getPos(access::location::gpu, access::mode::read).raw();

            computational.box = gd->getEnsemble()->getBox();
	    computational.lambda = gd->getEnsemble()->getLambda();
	    computational.paramPairIterator = storage.Param->getPairIterator();

            computational.ELECOEF = storage.ELECOEF;

            computational.dielectricConstant = storage.dielectricConstant;
            computational.debyeLength = storage.debyeLength;
            computational.alpha = storage.alpha;
            computational.n = storage.n;

            computational.cutOffFactor = storage.cutOffFactor;
            computational.cutOff = storage.cutOff;

            return computational;
        }

        //Storage data reader

        static __host__ StorageData getStorageData(std::shared_ptr<GlobalData>    gd,
                                                   std::shared_ptr<ParticleGroup> pg,
                                                   DataEntry& data){

            StorageData storage;

            storage.ELECOEF = gd->getUnits()->getElectricConversionFactor();

            storage.dielectricConstant = data.getParameter<real>("dielectricConstant");
            storage.debyeLength        = data.getParameter<real>("debyeLength");
            storage.alpha              = data.getParameter<real>("alpha");
            storage.n                  = data.getParameter<real>("n");
	    storage.Param = std::make_shared<ParameterPairsHandler>(gd,pg,data);

            storage.cutOffFactor = data.getParameter<real>("cutOffFactor");
            storage.cutOff       = storage.cutOffFactor*storage.debyeLength;

	    auto pairsParam = storage.Param->getPairParameters();
	    real maxSigma = 0.0;
            for(auto p : pairsParam){
                maxSigma=std::max(maxSigma,p.second.sigma);
            }
	    if (maxSigma > storage.cutOff)
	    {
		    storage.cutOff = maxSigma;
	    }

            System::log<System::MESSAGE>("[softWCA_DH] cutOff: %f" ,storage.cutOff);

            return storage;

        }


        static inline __device__ real energy(const int index_i,const int index_j,
                                             const ComputationalData& computational){

            const real4 posi = computational.pos[index_i];
            const real4 posj = computational.pos[index_j];

            const real3 rij = computational.box.apply_pbc(make_real3(posj)-make_real3(posi));
            const real r2   = dot(rij, rij);

            real e = real(0.0);

            real cutOff2 = computational.cutOff*computational.cutOff;
            const real chgProduct = computational.paramPairIterator(index_i,index_j).chargeProduct;
            if(r2>0 and r2<=cutOff2 and chgProduct != real(0.0)){

                e+=BasicPotentials::DebyeHuckel::DebyeHuckel::energy(rij,r2,
                                                                     computational.ELECOEF,
                                                                     chgProduct,
                                                                     computational.dielectricConstant,
                                                                     computational.debyeLength);
            }

	    const real epsilon  = computational.paramPairIterator(index_i,index_j).epsilon;
            const real sigma    = computational.paramPairIterator(index_i,index_j).sigma;
            const real lambda   = computational.lambda;
            const real alpha    = computational.alpha;
            const int n         = computational.n;
	    const real Acomodo  = alpha*(real(1.0)-lambda)*(real(1.0)-lambda); 
	    const real minPos   = sigma*pow(real(1.0) - Acomodo, real(1.0)/real(6.0));
	    const real minPos2  = minPos*minPos;	    //position of minimum
	    if (r2 < minPos2)
	    {
		    e += BasicPotentials::LennardJones::SoftCoreType2::energy(rij,r2,epsilon,sigma,lambda,alpha,n);
		    const real3 rij0 = make_real3(0.0);
		    e += -BasicPotentials::LennardJones::SoftCoreType2::energy(rij0,minPos2,epsilon,sigma,lambda,alpha,n);
	    }

            return e;
        }

      static inline __device__ real3 force(const int index_i,const int index_j,
                                             const ComputationalData& computational){

            const real4 posi = computational.pos[index_i];
            const real4 posj = computational.pos[index_j];

            const real3 rij = computational.box.apply_pbc(make_real3(posj)-make_real3(posi));
            const real r2   = dot(rij, rij);

            real3 f = make_real3(real(0.0));

            real cutOff2 = computational.cutOff*computational.cutOff;
            const real chgProduct = computational.paramPairIterator(index_i,index_j).chargeProduct;
            if(r2>0 and r2<=cutOff2 and chgProduct != real(0.0)){

                f+=BasicPotentials::DebyeHuckel::DebyeHuckel::force(rij,r2,
                                                                    computational.ELECOEF,
                                                                    chgProduct,
                                                                    computational.dielectricConstant,
                                                                    computational.debyeLength);
            }

	    const real epsilon = computational.paramPairIterator(index_i,index_j).epsilon;
            const real sigma   = computational.paramPairIterator(index_i,index_j).sigma;
            const real lambda  = computational.lambda;
            const real alpha   = computational.alpha;
            const int n        = computational.n;
	    const real Acomodo = alpha*(real(1.0)-lambda)*(real(1.0)-lambda); 
	    const real minPos  = sigma*pow(real(1.0) - Acomodo, real(1.0)/real(6.0));
	    const real minPos2 = minPos*minPos;	    //position of minimum
	    if (r2 < minPos2)
            {
                    f += BasicPotentials::LennardJones::SoftCoreType2::force(rij,r2,epsilon,sigma,lambda,alpha,n);
            }

            return f;
        }

      static inline __device__ tensor3 hessian(const int index_i,const int index_j,
                                               const ComputationalData& computational){

            const real4 posi = computational.pos[index_i];
            const real4 posj = computational.pos[index_j];

            const real3 rij = computational.box.apply_pbc(make_real3(posj)-make_real3(posi));
            const real r2   = dot(rij, rij);

            tensor3 H = tensor3(real(0.0));

            real cutOff2 = computational.cutOff*computational.cutOff;
            const real chgProduct = computational.paramPairIterator(index_i,index_j).chargeProduct;
            if(r2>0 and r2<=cutOff2 and chgProduct != real(0.0)){

                H+=BasicPotentials::DebyeHuckel::DebyeHuckel::hessian(rij,r2,
                                                                      computational.ELECOEF,
                                                                      chgProduct,
                                                                      computational.dielectricConstant,
                                                                      computational.debyeLength);
            }

            const real epsilon = computational.paramPairIterator(index_i,index_j).epsilon;
            const real sigma   = computational.paramPairIterator(index_i,index_j).sigma;
            const real lambda  = computational.lambda;
            const real alpha   = computational.alpha;
            const int n        = computational.n;
	    const real Acomodo = sigma*alpha*(real(1.0)-lambda)*(real(1.0)-lambda); 
	    const real minPos       = sigma*pow(real(1.0) - Acomodo, real(1.0)/real(6.0));
	    const real minPos2      = minPos*minPos;	    //position of minimum
	    if (r2 < minPos2)
            {
                    H += BasicPotentials::LennardJones::SoftCoreType2::hessian(rij,r2,epsilon,sigma,lambda,alpha,n);
            }

            return H;
        }

      static inline __device__ real lambdaDerivative(int index_i, int index_j,
                                                       const ComputationalData& computational){

            const real4 posi = computational.pos[index_i];
            const real4 posj = computational.pos[index_j];

            const real3 rij = computational.box.apply_pbc(make_real3(posj)-make_real3(posi));

            const real epsilon = computational.paramPairIterator(index_i,index_j).epsilon;
            const real sigma   = computational.paramPairIterator(index_i,index_j).sigma;
            const real lambda  = computational.lambda;
            const real alpha   = computational.alpha;
            const int  n       = computational.n;

            const real r2 = dot(rij, rij);
	    const real Acomodo = sigma*alpha*(real(1.0)-lambda)*(real(1.0)-lambda); 
	    const real minPos       = sigma*pow(real(1.0) - Acomodo, real(1.0)/real(6.0));
	    const real minPos2      = minPos*minPos;	    //position of minimum

            real ld = real(0.0);

            if (r2 < minPos2)
	    {
                ld = BasicPotentials::LennardJones::SoftCoreType2::lambdaDerivative(rij,r2,epsilon,sigma,lambda,alpha,n);
            }

            return ld;

        }
    };

    using softWCA_DH = NonBondedHessian_<softWCA_DH_>;

}}}}

REGISTER_NONBONDED_INTERACTOR(
    NonBonded,softWCA_DH,
    uammd::structured::Interactor::PairInteractor<uammd::structured::Potentials::NonBonded::softWCA_DH>
)
