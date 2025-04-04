#include "System/ExtendedSystem.cuh"
#include "GlobalData/GlobalData.cuh"
#include "ParticleData/ExtendedParticleData.cuh"
#include "ParticleData/ParticleGroup.cuh"
#include "ParticleGroup/ParticleGroupUtils.cuh"

#include "Integrator/IntegratorBase.cuh"
#include "Integrator/IntegratorFactory.cuh"
#include "Integrator/IntegratorUtils.cuh"

#include <fstream>
#include <unistd.h>
#include "Utils/Measures/MeasuresBasic.cuh"

namespace uammd{
namespace structured{
namespace Integrator{
namespace Special{
namespace Rerun{

	class  Rerun: public IntegratorBase
	{

		private:
			std::string inputTrajectory;
			bool initialized = false;
			std::ifstream leggi;

			thrust::device_vector<real4> posRef;
			void copyToRef() //from host (cpu) to device (gpu)
			{
	    			auto pos = this->pd->getPos(access::location::gpu, access::mode::readwrite);
	    			posRef.resize(this->pd->getNumParticles());
	    			thrust::copy(thrust::cuda::par.on(stream),pos.begin(), pos.end(), posRef.begin());

	    			CudaSafeCall(cudaStreamSynchronize(stream));
			}
			void copyFromRef() //from device (gpu) to host (cpu)
			{
	    			auto pos = this->pd->getPos(access::location::gpu, access::mode::readwrite);
	    			thrust::copy(thrust::cuda::par.on(stream),posRef.begin(), posRef.end(), pos.begin());

	    			CudaSafeCall(cudaStreamSynchronize(stream));
		    	}
		public:
		  	Rerun(std::shared_ptr<GlobalData> gd, std::shared_ptr<ParticleGroup> pg, DataEntry& data, std::string name):IntegratorBase(gd,pg,data,name)
			{
		  		System::log<System::MESSAGE>("[Rerun] Created Rerun integrator \"%s\"",name.c_str());
		  		inputTrajectory   = data.getParameter<std::string>("inputTrajectory");
			}

			int init()
			{
	    			System::log<System::MESSAGE>("[Rerun] Performing initialization step");

                    		//refPos(0)
	    			this->copyToRef();

                    		//Reset force to 0 and compute force
		                this->resetForce();
	    			this->updateForce();
	    			CudaSafeCall(cudaStreamSynchronize(stream));
				CudaCheckError();

				//Open stream 
				int flag;
				if (::access(inputTrajectory.c_str(), F_OK ) == -1 )
				{
					System::log<System::CRITICAL>("[Rerun] File %s not found", inputTrajectory.c_str());
				}
				leggi.open(inputTrajectory.c_str(), std::ifstream::in);
				flag = ReadDump();
				flag = ReadDump();

				initialized = true;
			}

			int ReadDump()
			{
				std::string comstr;
				getline(leggi, comstr);
				real comodo;
				int indice, tipo, natoms;
				leggi >> comodo;
				printf("checkpoint 1\n");
				if (!leggi.eof())
				{
					this->gd->getFundamental()->setSimulationTime(comodo); // timestep
					getline(leggi, comstr);
					getline(leggi, comstr);
					leggi >> natoms; // # of atoms
					int N = this->pg->getNumberParticles();
					if (natoms != N)
					{
						System::log<System::CRITICAL>("[Rerun] # of atoms in file (%d) is distinct from # of atoms in simulation (%d)", natoms, N);
					}
					getline(leggi, comstr);
					getline(leggi, comstr);
					printf("checkpoint 2 %d\n", natoms);

					// box info
					Box box = this->gd->getEnsemble()->getBox();
					real box_left, box_right, box_size;
					leggi >> box_left >> box_right;
					box_size = box_right - box_left;
					box.boxSize.x = box_size/real(2.0);
					leggi >> box_left >> box_right;
					box_size = box_right - box_left;
					box.boxSize.y = box_size/real(2.0);
					leggi >> box_left >> box_right;
					box_size = box_right - box_left;
					box.boxSize.z = box_size/real(2.0);
					this->gd->getEnsemble()->setBox(box);
					getline(leggi, comstr);
					getline(leggi, comstr);
					printf("checkpoint 3\n");

					// coordinates
					real4* posRef_ptr  = thrust::raw_pointer_cast(posRef.data());
					std::vector <real> pos_loc_x(natoms), pos_loc_y(natoms), pos_loc_z(natoms);
					for (int i=0; i<natoms; i++)
					{
						leggi >> comodo >> comodo >> pos_loc_x[i] >> pos_loc_y[i] >> pos_loc_z[i];
					}
                        		CudaSafeCall(cudaStreamSynchronize(stream));
					CudaCheckError();

					{
						auto pos = this->pd->getPos(access::location::gpu, access::mode::readwrite);						
						auto groupIterator = this->pg->getIndexIterator(access::location::gpu);
						real4* pos_ptr = pos.raw();
						thrust::for_each(thrust::cuda::par.on(stream),groupIterator,groupIterator + N, [=] __host__ __device__ (int index)
						{
							pos_ptr[index].x = pos_loc_x[index];
							pos_ptr[index].y = pos_loc_y[index];
							pos_ptr[index].z = pos_loc_z[index];
						});
					}
                        		CudaSafeCall(cudaStreamSynchronize(stream));
					CudaCheckError();
					//copyFromRef();
                        		//CudaSafeCall(cudaStreamSynchronize(stream));
					//CudaCheckError();
					//printf("CIAO2\n");
	    				//auto pos = this->pd->getPos(access::location::gpu, access::mode::readwrite);
					////for (int index=0; index<natoms; index++)
					////	//printf("CIAONE2 %d %f %f %f\n", index, posRef_ptr[index].x, posRef_ptr[index].y, posRef_ptr[index].z);
					////	printf("CIAONE2 %d %f %f %f\n", index, pos[index].x, pos[index].y, pos[index].z);
					
					getline(leggi, comstr);
                        		CudaSafeCall(cudaStreamSynchronize(stream));
					CudaCheckError();
					
					printf("checkpoint 4 %f %s\n", pos_loc_z[natoms-1], comstr.c_str());

					return 1;
				}
				else
				{
					return 0;
				}
			}

			void forwardTime() override 
			{
				int flag; 
				if(not initialized)
				{
					flag = this->init();
	    			}
				this->updateForce();
	    			this->integrationStep();
				this->gd->getFundamental()->setCurrentStep(this->gd->getFundamental()->getCurrentStep()+1);
				//this->gd->getFundamental()->setSimulationTime(this->gd->getFundamental()->getSimulationTime()+this->dt);
			}

			void integrationStep()
			{
				int flag = ReadDump();
			}

	};

}}}}}

REGISTER_INTEGRATOR(
    Rerun,Rerun,
    uammd::structured::Integrator::Special::Rerun::Rerun
)
