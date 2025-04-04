#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"

#include "Integrator/IntegratorBase.cuh"
#include "Integrator/IntegratorFactory.cuh"
#include "Integrator/IntegratorUtils.cuh"

namespace uammd{
namespace structured{
namespace Integrator{
namespace Special{
namespace Rerun{

	class Rerun : public IntegratorBaseNVT{

		private:

			std::ifstream readerInputTrajectory;
			std::string inputTrajectory;
			thrust::device_vector <real> posLocX_d, posLocY_d, posLocZ_d;
			thrust::host_vector <real> posLocX_h, posLocY_h, posLocZ_h;
			bool firstStep = true;

		public:

      Rerun(std::shared_ptr<GlobalData>           gd,
                     std::shared_ptr<ParticleGroup>        pg,
                     DataEntry& data,
                     std::string name):IntegratorBaseNVT(gd,pg,data,name){
	      			inputTrajectory = data.getParameter<std::string>("inputTrajectory");
				uint N = this->pg->getNumberParticles();
				posLocX_d.resize(N);
				posLocY_d.resize(N);
				posLocZ_d.resize(N);
				posLocX_h.resize(N);
				posLocY_h.resize(N);
				posLocZ_h.resize(N);

				readerInputTrajectory.open(inputTrajectory.c_str(), std::ifstream::in);
				System::log<System::MESSAGE>("[Rerun] (%s) Opened stream for input trajectory %s",name.c_str(), inputTrajectory.c_str());
				
				System::log<System::DEBUG1>("[Rerun] (%s) Performing rerun of step %llu",name.c_str(), this->gd->getFundamental()->getCurrentStep());
				this->integrationStep();
			}

			void forwardTime() override {

				if(firstStep){
					firstStep = false;
				}

				System::log<System::DEBUG1>("[Rerun] (%s) Performing rerun of step %llu",name.c_str(), this->gd->getFundamental()->getCurrentStep());
				//this->updateForce();
				//CudaSafeCall(cudaStreamSynchronize(stream));
				//CudaCheckError();
				real currentTempo = this->integrationStep();
				printf("IGNAZIO %f\n", currentTempo);
				this->gd->getFundamental()->setCurrentStep(this->gd->getFundamental()->getCurrentStep()+1);
				//this->gd->getFundamental()->setSimulationTime(this->gd->getFundamental()->getSimulationTime()+this->dt);
				this->gd->getFundamental()->setSimulationTime(currentTempo);
			}

			real integrationStep(){
				uint N           = this->pg->getNumberParticles();
				auto pos         = this->pd->getPos(access::location::cpu, access::mode::readwrite);
				real4* pos_ptr   = pos.raw();

				std::string localString; 				
				real passoTempo;

				getline(readerInputTrajectory, localString); //skip first line
				//printf("IGNAZIO %s\n", localString.c_str());
                                readerInputTrajectory >> passoTempo;
				//printf("IGNAZIO %12.10f\n", passoTempo);
                                for (int i=2; i<=9; i++) // skip rest of the header
				{
                                        getline(readerInputTrajectory, localString);
					//printf("IGNAZIO %s\n", localString.c_str());
				}
                                int iLoc, tipo;
                                for (int i=0; i<N; i++)
                                {
                                        readerInputTrajectory >> iLoc >> tipo >> pos_ptr[i].x >> pos_ptr[i].y >> pos_ptr[i].z;
					//printf("IGNAZIO %12.10f %d %d %12.10f %12.10f %12.10f\n", passoTempo, iLoc, tipo, pos_ptr[i].x, pos_ptr[i].y, pos_ptr[i].z);
                                }
				//printf("IGNAZIO %12.10f %d %d %12.10f %12.10f %12.10f\n", passoTempo, iLoc, tipo, pos_ptr[N-1].x, pos_ptr[N-1].y, pos_ptr[N-1].z);
                                getline(readerInputTrajectory, localString); // close the last line

				return passoTempo;
			}

	};

}}}}}

REGISTER_INTEGRATOR(
    Rerun,Rerun,
    uammd::structured::Integrator::Special::Rerun::Rerun
)
