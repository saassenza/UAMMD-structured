#include "System/ExtendedSystem.cuh"
#include "Simulation/Simulation.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"

#include "Integrator/IntegratorBase.cuh"
#include "Integrator/IntegratorFactory.cuh"
#include "Integrator/IntegratorUtils.cuh"
#include <unistd.h>

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

				if (::access(inputTrajectory.c_str(), F_OK ) == -1 )
                                {
                                        System::log<System::CRITICAL>("[Rerun] File %s not found", inputTrajectory.c_str());
                                }
				readerInputTrajectory.open(inputTrajectory.c_str(), std::ifstream::in);
				std::string localString; 				
				getline(readerInputTrajectory, localString); //skip first line
				getline(readerInputTrajectory, localString); //skip second line
				getline(readerInputTrajectory, localString); //skip third line
				int nInput;
				readerInputTrajectory >> nInput;
				if (nInput != N)
				{
					System::log<System::CRITICAL>("[Rerun] Number of particles %d in file %s is different than input value %d", nInput, inputTrajectory.c_str(), N);
				}
				readerInputTrajectory.close();
				
				System::log<System::MESSAGE>("[Rerun] (%s) Opened stream for input trajectory %s",name.c_str(), inputTrajectory.c_str());
				readerInputTrajectory.open(inputTrajectory.c_str(), std::ifstream::in);
				
				System::log<System::DEBUG1>("[Rerun] (%s) Performing rerun of step %llu",name.c_str(), this->gd->getFundamental()->getCurrentStep());
				this->integrationStep();
			}

			void forwardTime() override {

				if(firstStep){
					firstStep = false;
				}

				System::log<System::DEBUG1>("[Rerun] (%s) Performing rerun of step %llu",name.c_str(), this->gd->getFundamental()->getCurrentStep());
				real currentTime = this->integrationStep();
				this->gd->getFundamental()->setCurrentStep(this->gd->getFundamental()->getCurrentStep()+1);
				this->gd->getFundamental()->setSimulationTime(currentTime);
			}

			real integrationStep(){
				uint N           = this->pg->getNumberParticles();
				auto pos         = this->pd->getPos(access::location::cpu, access::mode::readwrite);
				real4* pos_ptr   = pos.raw();

				std::string localString; 				
				real currentTime;

				getline(readerInputTrajectory, localString); //skip first line
                                readerInputTrajectory >> currentTime;
                                for (int i=2; i<=9; i++) // skip rest of the header
				{
                                        getline(readerInputTrajectory, localString);
				}
                                int iLoc, tipo;
                                for (int i=0; i<N; i++)
                                {
                                        readerInputTrajectory >> iLoc >> tipo >> pos_ptr[i].x >> pos_ptr[i].y >> pos_ptr[i].z;
                                }
                                getline(readerInputTrajectory, localString); // close the last line

				return currentTime;
			}

	};

}}}}}

REGISTER_INTEGRATOR(
    Rerun,Rerun,
    uammd::structured::Integrator::Special::Rerun::Rerun
)
