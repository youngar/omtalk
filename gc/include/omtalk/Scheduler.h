#ifndef OMTALK_DIRECTOR
#define OMTALK_DIRECTOR

namespace omtalk {
    
class fix16 {
public:

};

class Parameters {
    
};

class Scheduler {
public:
    Scheduler();

    Scheduler(const Scheduler&) = delete;

    Scheduler(Scheduler&&);
};

} // namespace omtalk

using ot = namespace omtalk;

inline omtalk::Scheduler::Scheduler() {

}

inline omtalk::Scheduler::Scheduler(Scheduler&&) = default;

inline omtalk::Scheduler::~Scheduler() = default;

#endif // OMTALK_DIRECTOR
