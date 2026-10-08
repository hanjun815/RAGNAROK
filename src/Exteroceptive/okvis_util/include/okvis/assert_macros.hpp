/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file assert_macros.hpp
 * @brief This file contains some useful assert macros.
 * @author Paul Furgale
 * @author Stefan Leutenegger
 */

#ifndef OKVIS_ASSERT_MACROS_HPP
#define OKVIS_ASSERT_MACROS_HPP

#include <sstream>
#include "okvis/source_file_pos.hpp"

//! Macro for defining an exception with a given parent
//  (std::runtime_error should be top parent)
// adapted from ros/drivers/laser/hokuyo_driver/hokuyo.h
#define OKVIS_DEFINE_EXCEPTION(exceptionName, exceptionParent)				\
  class exceptionName : public exceptionParent {						\
  public:																\
  typedef __typeof__(exceptionParent) exceptionParent_t;  \
  using exceptionParent_t::exceptionParent_t; \
};

/// \brief okvis Main namespace of this package.
namespace okvis {

  namespace detail {

    template<typename OKVIS_EXCEPTION_T>
    inline void OKVIS_throw_exception(std::string const & exceptionType, okvis::source_file_pos sfp, std::string const & message)
    {
      std::stringstream okvis_assert_stringstream;
      // I have no idea what broke doesn't work with the << operator. sleutenegger: not just Windows, but in general...???
      okvis_assert_stringstream << exceptionType <<  sfp.toString() << " " << message;
      throw(OKVIS_EXCEPTION_T(okvis_assert_stringstream.str()));
    }

    template<typename OKVIS_EXCEPTION_T>
    inline void OKVIS_throw_exception(std::string const & exceptionType, std::string const & function, std::string const & file,
								   int line, std::string const & message)
    {
      OKVIS_throw_exception<OKVIS_EXCEPTION_T>(exceptionType, okvis::source_file_pos(function,file,line),message);
    }


  } // namespace okvis::detail

  template<typename OKVIS_EXCEPTION_T>
  inline void okvis_assert_throw(bool assert_condition, std::string message, okvis::source_file_pos sfp) {
    if(!assert_condition)
      {
		detail::OKVIS_throw_exception<OKVIS_EXCEPTION_T>("", sfp,message);
      }
  }



} // namespace okvis

#define OKVIS_CHECK_MAP(obj, id)                                                                                          	\
  if(!(obj.count(id)))															\
  {																	\
      std::stringstream okvis_assert_stringstream;											\
      okvis_assert_stringstream << #obj << ".at(" << #id <<") failed! "; 						\
      okvis::detail::OKVIS_throw_exception<std::runtime_error>("[OKVIS_AT_CHECKED] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
  }

#define OKVIS_THROW(exceptionType, message) {								\
    std::stringstream okvis_assert_stringstream;							\
    okvis_assert_stringstream << message;									\
    okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__, okvis_assert_stringstream.str()); \
  }


#define OKVIS_THROW_SFP(exceptionType, SourceFilePos, message){			\
    std::stringstream okvis_assert_stringstream;							\
    okvis_assert_stringstream << message;									\
    okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", SourceFilePos, okvis_assert_stringstream.str()); \
  }

#define OKVIS_ASSERT_TRUE(exceptionType, condition, message)				\
  if(!(condition))														\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "assert(" << #condition << ") failed: " << message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__, okvis_assert_stringstream.str()); \
    }

#define OKVIS_ASSERT_FALSE(exceptionType, condition, message)				\
  if((condition))														\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "assert( not " << #condition << ") failed: " << message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__, okvis_assert_stringstream.str()); \
    }



#define OKVIS_ASSERT_GE_LT(exceptionType, value, lowerBound, upperBound, message) \
  if((value) < (lowerBound) || (value) >= (upperBound))							\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "assert(" << #lowerBound << " <= " << #value << " < " << #upperBound << ") failed [" << (lowerBound) << " <= " << (value) << " < " << (upperBound) << "]: " << message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }



#define OKVIS_ASSERT_LT(exceptionType, value, upperBound, message)			\
  if((value) >= (upperBound))												\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "assert(" << #value << " < " << #upperBound << ") failed [" << (value) << " < " << (upperBound) << "]: " <<  message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }

#define OKVIS_ASSERT_GE(exceptionType, value, lowerBound, message)			\
  if((value) < (lowerBound))												\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "assert(" << #value << " >= " << #lowerBound << ") failed [" << (value) << " >= " << (lowerBound) << "]: " <<  message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }



#define OKVIS_ASSERT_LE(exceptionType, value, upperBound, message)			\
  if((value) > (upperBound))												\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "assert(" << #value << " <= " << #upperBound << ") failed [" << (value) << " <= " << (upperBound) << "]: " <<  message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }

#define OKVIS_ASSERT_GT(exceptionType, value, lowerBound, message)			\
  if((value) <= (lowerBound))												\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "assert(" << #value << " > " << #lowerBound << ") failed [" << (value) << " > " << (lowerBound) << "]: " <<  message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }



#define OKVIS_ASSERT_EQ(exceptionType, value, testValue, message)			\
  if((value) != (testValue))												\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "assert(" << #value << " == " << #testValue << ") failed [" << (value) << " == " << (testValue) << "]: " <<  message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }

#define OKVIS_ASSERT_NE(exceptionType, value, testValue, message)			\
  if((value) == (testValue))												\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "assert(" << #value << " != " << #testValue << ") failed [" << (value) << " != " << (testValue) << "]: " <<  message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }

#define OKVIS_ASSERT_NEAR(exceptionType, value, testValue, abs_error, message) \
  if(!(fabs((testValue) - (value)) <= fabs(abs_error)))						\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "assert(" << #value << " == " << #testValue << ") failed [" << (value) << " == " << (testValue) << " (" << fabs((testValue) - (value)) << " > " << fabs(abs_error) << ")]: " <<  message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }



#ifndef NDEBUG

#define OKVIS_THROW_DBG(exceptionType, message){							\
    std::stringstream okvis_assert_stringstream;							\
    okvis_assert_stringstream << message;									\
    okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__, okvis_assert_stringstream.str()); \
  }



#define OKVIS_ASSERT_TRUE_DBG(exceptionType, condition, message)			\
  if(!(condition))														\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "debug assert(" << #condition << ") failed: " << message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__, okvis_assert_stringstream.str()); \
    }

#define OKVIS_ASSERT_FALSE_DBG(exceptionType, condition, message)			\
  if((condition))														\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "debug assert( not " << #condition << ") failed: " << message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__, okvis_assert_stringstream.str()); \
    }


#define OKVIS_ASSERT_DBG_RE( condition, message) OKVIS_ASSERT_DBG(std::runtime_error, condition, message)

#define OKVIS_ASSERT_GE_LT_DBG(exceptionType, value, lowerBound, upperBound, message) \
  if((value) < (lowerBound) || (value) >= (upperBound))							\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "debug assert(" << #lowerBound << " <= " << #value << " < " << #upperBound << ") failed [" << (lowerBound) << " <= " << (value) << " < " << (upperBound) << "]: " << message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }



#define OKVIS_ASSERT_LT_DBG(exceptionType, value, upperBound, message)		\
  if((value) >= (upperBound))												\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "debug assert(" << #value << " < " << #upperBound << ") failed [" << (value) << " < " << (upperBound) << "]: " <<  message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }



#define OKVIS_ASSERT_GE_DBG(exceptionType, value, lowerBound, message)		\
  if((value) < (lowerBound))												\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "debug assert(" << #value << " >= " << #lowerBound << ") failed [" << (value) << " >= " << (lowerBound) << "]: " <<  message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }



#define OKVIS_ASSERT_LE_DBG(exceptionType, value, upperBound, message)		\
  if((value) > (upperBound))												\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "debug assert(" << #value << " <= " << #upperBound << ") failed [" << (value) << " <= " << (upperBound) << "]: " <<  message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }

#define OKVIS_ASSERT_GT_DBG(exceptionType, value, lowerBound, message)		\
  if((value) <= (lowerBound))												\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "debug assert(" << #value << " > " << #lowerBound << ") failed [" << (value) << " > " << (lowerBound) << "]: " <<  message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }



#define OKVIS_ASSERT_EQ_DBG(exceptionType, value, testValue, message)		\
  if((value) != (testValue))												\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "debug assert(" << #value << " == " << #testValue << ") failed [" << (value) << " == " << (testValue) << "]: " <<  message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }


#define OKVIS_ASSERT_NE_DBG(exceptionType, value, testValue, message)		\
  if((value) == (testValue))												\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "debug assert(" << #value << " != " << #testValue << ") failed [" << (value) << " != " << (testValue) << "]: " <<  message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }



#define OKVIS_ASSERT_NEAR_DBG(exceptionType, value, testValue, abs_error, message) \
  if(!(fabs((testValue) - (value)) <= fabs(abs_error)))						\
    {																	\
      std::stringstream okvis_assert_stringstream;							\
      okvis_assert_stringstream << "debug assert(" << #value << " == " << #testValue << ") failed [" << (value) << " == " << (testValue) << " (" << fabs((testValue) - (value)) << " > " << fabs(abs_error) << ")]: " <<  message; \
      okvis::detail::OKVIS_throw_exception<exceptionType>("[" #exceptionType "] ", __FUNCTION__,__FILE__,__LINE__,okvis_assert_stringstream.str()); \
    }


#define OKVIS_OUT(X) std::cout << #X << ": " << (X) << std::endl

#else

#define OKVIS_OUT(X)
#define OKVIS_THROW_DBG(exceptionType, message)
#define OKVIS_ASSERT_TRUE_DBG(exceptionType, condition, message)
#define OKVIS_ASSERT_FALSE_DBG(exceptionType, condition, message)
#define OKVIS_ASSERT_GE_LT_DBG(exceptionType, value, lowerBound, upperBound, message)
#define OKVIS_ASSERT_LT_DBG(exceptionType, value, upperBound, message)
#define OKVIS_ASSERT_GT_DBG(exceptionType, value, lowerBound, message)
#define OKVIS_ASSERT_LE_DBG(exceptionType, value, upperBound, message)
#define OKVIS_ASSERT_GE_DBG(exceptionType, value, lowerBound, message)
#define OKVIS_ASSERT_NE_DBG(exceptionType, value, testValue, message)
#define OKVIS_ASSERT_EQ_DBG(exceptionType, value, testValue, message)
#define OKVIS_ASSERT_NEAR_DBG(exceptionType, value, testValue, abs_error, message)	
#endif



#endif // OKVIS_ASSERT_MACROS_HPP

