#ifndef __INTEGRATOR_BDHI_OPEN_BOUNDARY_CHOLESKY__
#define __INTEGRATOR_BDHI_OPEN_BOUNDARY_CHOLESKY__

#include"Integrator/BDHI/BDHI_EulerMaruyama.cuh"
#include"Integrator/BDHI/BDHI_Cholesky.cuh"

namespace uammd{
namespace structured{
namespace NVT{
namespace BDHIOpenBoundary{

	class Cholesky : public IntegratorBasicNVT{

		private:

			using BDHI = BDHI::EulerMaruyama<BDHI::Cholesky>;

			std::unique_ptr<BDHI> bdhi;
			bool firstStep = true;

		public:

      Cholesky(std::shared_ptr<GlobalData>           gd,
               std::shared_ptr<ParticleGroup>        pg,
               DataEntry& data,
               std::string name):IntegratorBasicNVT(gd,pg,data,name){

				int batchNumber = GroupUtils::BatchGroupNumber(pg);
				if(batchNumber > 1){
					System::log<System::CRITICAL>("[BDHI] This integrator can not handle more than one batch.");
				}

				BDHI::Parameters bdhiParameters;

				bdhiParameters.dt = this->dt;

				bdhiParameters.temperature = this->kBT;
				bdhiParameters.viscosity = data.getParameter<real>("viscosity");

				bdhiParameters.hydrodynamicRadius = data.getParameter<real>("hydrodynamicRadius",-1.0);

				System::log<System::MESSAGE>("[BDHI] Viscosity: %f",bdhiParameters.viscosity);

				if(bdhiParameters.hydrodynamicRadius < 0.0){
					System::log<System::MESSAGE>("[BDHI] Hydrodynamic radius not set, using particle radius");
				} else {
					System::log<System::MESSAGE>("[BDHI] Hydrodynamic radius: %f",bdhiParameters.hydrodynamicRadius);
				}

				bdhi = std::make_unique<BDHI>(pg,bdhiParameters);

			}

			void forwardTime() override {

				if(firstStep){
					//Load all interactors into bdhi
					for(auto& interactor : this->getInteractors()){
						bdhi->addInteractor(interactor);
					}

					//Load all updatables into bdhi
					for(auto& updatable : this->getUpdatables()){
						bdhi->addUpdatable(updatable);
					}

					firstStep = false;
				}

				bdhi->forwardTime();
        this->gd->getFundamental()->setCurrentStep(this->gd->getFundamental()->getCurrentStep()+1);
        this->gd->getFundamental()->setSimulationTime(this->gd->getFundamental()->getSimulationTime()+this->dt);
			}

	};

}}}}

#endif
