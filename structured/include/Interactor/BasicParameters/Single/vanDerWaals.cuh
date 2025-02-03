#pragma once

namespace uammd{
namespace structured{
namespace Potentials{
namespace BasicParameters{

    namespace Single{

        struct vanDerWaals{

            struct InputSingleParameters{

                std::string name;

                real WCA_epsilon;
                real particleSqrtHamakerOverDensity;
            };

            struct SingleParameters{
                real WCA_epsilon;
                real particleSqrtHamakerOverDensity;
            };

            template<typename T>
            static inline __host__ InputSingleParameters readSingleParameters(std::map<std::string,T>& info){

                InputSingleParameters param;

                param.name    = std::string(info.at("name"));
                param.WCA_epsilon = real(info.at("WCA_epsilon"));
                param.particleSqrtHamakerOverDensity   = real(info.at("particleSqrtHamakerOverDensity"));

                return param;
            }

            static inline __host__ SingleParameters processSingleParameters(InputSingleParameters in_par){
                SingleParameters param;

                param.WCA_epsilon = in_par.WCA_epsilon;
                param.particleSqrtHamakerOverDensity   = in_par.particleSqrtHamakerOverDensity;

                return param;

            }
        };

    }

}}}}
