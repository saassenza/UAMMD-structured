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

    struct WCA_DH_{

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

	    real ELECOEF;
	    real debyeLength;
	    real dielectricConstant;
        };

        //Potential parameters
        struct StorageData{

            std::shared_ptr<ParameterPairsHandler> WCAParam;

            real cutOffFactor;
            real cutOff;

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

            computational.paramPairIterator = storage.WCAParam->getPairIterator();
	    
	    computational.cutOff = storage.cutOff;

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
            
	    storage.WCAParam = std::make_shared<ParameterPairsHandler>(gd,pg,data);

            storage.cutOffFactor  = data.getParameter<real>("cutOffFactor");
            
	    storage.debyeLength = data.getParameter<real>("debyeLength");;
	    storage.ELECOEF = gd->getUnits()->getElectricConversionFactor();
	    storage.dielectricConstant = data.getParameter<real>("dielectricConstant");;

            ///////////////////////////////////////////////////////////


            ///////////////////////////////////////////////////////////

            auto pairsParam = storage.WCAParam->getPairParameters();

            real maxSigma = 0.0;
            for(auto p : pairsParam){
                maxSigma=std::max(maxSigma,p.second.sigma);
            }

            storage.cutOff = maxSigma*storage.cutOffFactor;

            System::log<System::MESSAGE>("[LennardJones] cutOff: %f" ,storage.cutOff);

            return storage;
        }

        static inline __device__ real energy(int index_i, int index_j,
                                             const ComputationalData& computational){

            const real4 posi = computational.pos[index_i];
            const real4 posj = computational.pos[index_j];

            const real3 rij = computational.box.apply_pbc(make_real3(posj)-make_real3(posi));

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
                e += prefactorDH*exp(-dist/lD)/dist;
            }

            // WCA
            if (r2 < sigma*sigma)
            {
		    real invRnorm2 = sigma*sigma/r2;
		    real invRnorm6 = invRnorm2*invRnorm2*invRnorm2;
                    e += epsilon*(invRnorm6*invRnorm6 - real(2.0)*invRnorm6 + real(1.0)); 
            }

            return e;

        }


        static inline __device__ real3 force(int index_i, int index_j,
                                             const ComputationalData& computational){

            const real4 posi = computational.pos[index_i];
            const real4 posj = computational.pos[index_j];

            const real3 rij = computational.box.apply_pbc(make_real3(posj)-make_real3(posi));
            
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
                real invDist = real(1.0)/(dist);
                real fmod = -prefactorDH*exp(-dist*invLD)/r2*(invLD + invDist);

                f += fmod*rij;
            }

            // WCA
            if (r2 < sigma*sigma)
            {
		    real invRnorm2 = sigma*sigma/r2;
		    real invRnorm6 = invRnorm2*invRnorm2*invRnorm2;
		    real fmod = -real(12.0)*epsilon/r2*(invRnorm6*invRnorm6 - invRnorm6);
    
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

    using WCA_DH = NonBondedHessian_<WCA_DH_>;

}}}}

REGISTER_NONBONDED_INTERACTOR(
    NonBonded,WCA_DH,
    uammd::structured::Interactor::PairInteractor<uammd::structured::Potentials::NonBonded::WCA_DH>
)
