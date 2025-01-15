//
// Copyright (c) 2019-2021 INRIA
//

#ifndef __pinocchio_multibody_joint_mimic_hpp__
#define __pinocchio_multibody_joint_mimic_hpp__

#include "pinocchio/multibody/joint/fwd.hpp"
#include "pinocchio/multibody/joint/joint-collection.hpp"
#include "pinocchio/macros.hpp"
#include "pinocchio/multibody/joint/joint-base.hpp"
#include "pinocchio/multibody/joint/joint-basic-visitors.hpp"
#include <boost/variant.hpp>
#include <boost/mpl/filter_view.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/mpl/placeholders.hpp>
#include <iostream>

namespace pinocchio
{
  template<typename _Scalar, int _Options, int MaxDim>
  struct ScaledJointMotionSubspaceTpl;

  template<typename _Scalar, int _Options, int _MaxDim>
  struct traits<ScaledJointMotionSubspaceTpl<_Scalar, _Options, _MaxDim>>
  {
    enum
    {
      MaxDim = _MaxDim
    };
    typedef JointMotionSubspaceTpl<Eigen::Dynamic, _Scalar, _Options, _MaxDim>
      RefJointMotionSubspace;
    typedef typename traits<RefJointMotionSubspace>::Scalar Scalar;
    enum
    {
      Options = traits<RefJointMotionSubspace>::Options
    };
    enum
    {
      LINEAR = traits<RefJointMotionSubspace>::LINEAR,
      ANGULAR = traits<RefJointMotionSubspace>::ANGULAR
    };
    typedef typename traits<RefJointMotionSubspace>::JointMotion JointMotion;
    typedef typename traits<RefJointMotionSubspace>::JointForce JointForce;
    typedef typename traits<RefJointMotionSubspace>::DenseBase DenseBase;
    typedef typename traits<RefJointMotionSubspace>::MatrixReturnType MatrixReturnType;
    typedef typename traits<RefJointMotionSubspace>::ConstMatrixReturnType ConstMatrixReturnType;
  }; // traits ScaledJointMotionSubspaceTpl

  template<typename _Scalar, int _Options, int _MaxDim>
  struct SE3GroupAction<ScaledJointMotionSubspaceTpl<_Scalar, _Options, _MaxDim>>
  {
    typedef typename SE3GroupAction<typename traits<
      ScaledJointMotionSubspaceTpl<_Scalar, _Options, _MaxDim>>::RefJointMotionSubspace>::ReturnType
      ReturnType;
  };

  template<typename _Scalar, int _Options, int _MaxDim, typename MotionDerived>
  struct MotionAlgebraAction<
    ScaledJointMotionSubspaceTpl<_Scalar, _Options, _MaxDim>,
    MotionDerived>
  {
    typedef typename MotionAlgebraAction<
      typename traits<
        ScaledJointMotionSubspaceTpl<_Scalar, _Options, _MaxDim>>::RefJointMotionSubspace,
      MotionDerived>::ReturnType ReturnType;
  };

  template<typename _Scalar, int _Options, int _MaxDim, typename ForceDerived>
  struct ConstraintForceOp<ScaledJointMotionSubspaceTpl<_Scalar, _Options, _MaxDim>, ForceDerived>
  {
    typedef
      typename ScaledJointMotionSubspaceTpl<_Scalar, _Options, _MaxDim>::RefJointMotionSubspace
        RefJointMotionSubspace;
    typedef
      typename ConstraintForceOp<RefJointMotionSubspace, ForceDerived>::ReturnType RefReturnType;
    typedef typename ScalarMatrixProduct<_Scalar, RefReturnType>::type ReturnType;
  };

  template<typename _Scalar, int _Options, int _MaxDim, typename ForceSet>
  struct ConstraintForceSetOp<ScaledJointMotionSubspaceTpl<_Scalar, _Options, _MaxDim>, ForceSet>
  {
    typedef
      typename ScaledJointMotionSubspaceTpl<_Scalar, _Options, _MaxDim>::RefJointMotionSubspace
        RefJointMotionSubspace;
    typedef
      typename ConstraintForceSetOp<RefJointMotionSubspace, ForceSet>::ReturnType RefReturnType;
    typedef typename ScalarMatrixProduct<_Scalar, RefReturnType>::type ReturnType;
  };

  template<typename _Scalar, int _Options, int _MaxDim>
  struct ScaledJointMotionSubspaceTpl
  : JointMotionSubspaceBase<ScaledJointMotionSubspaceTpl<_Scalar, _Options, _MaxDim>>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PINOCCHIO_CONSTRAINT_TYPEDEF_TPL(ScaledJointMotionSubspaceTpl)
    enum
    {
      NV = Eigen::Dynamic,
      MaxDim = _MaxDim
    };
    typedef JointMotionSubspaceBase<ScaledJointMotionSubspaceTpl> Base;
    using Base::nv;

    typedef typename traits<ScaledJointMotionSubspaceTpl<_Scalar, _Options, _MaxDim>>::
      RefJointMotionSubspace RefJointMotionSubspace;
    typedef typename SE3GroupAction<RefJointMotionSubspace>::ReturnType SE3ActionReturnType;

    ScaledJointMotionSubspaceTpl()
    : ScaledJointMotionSubspaceTpl(1.0)
    {
    }

    explicit ScaledJointMotionSubspaceTpl(const Scalar & scaling_factor)
    : m_scaling_factor(scaling_factor)
    , m_constraint(0)
    {
    }

    template<typename ConstraintTpl>
    ScaledJointMotionSubspaceTpl(const ConstraintTpl & constraint, const Scalar & scaling_factor)
    : m_constraint(constraint)
    , m_scaling_factor(scaling_factor)
    {
    }

    ScaledJointMotionSubspaceTpl(const ScaledJointMotionSubspaceTpl & other)
    : m_constraint(other.m_constraint)
    , m_scaling_factor(other.m_scaling_factor)
    {
    }

    ScaledJointMotionSubspaceTpl & operator=(const ScaledJointMotionSubspaceTpl & other)
    {
      m_constraint = other.m_constraint;
      m_scaling_factor = other.m_scaling_factor;
      return *this;
    }

    template<typename VectorLike>
    JointMotion __mult__(const Eigen::MatrixBase<VectorLike> & v) const
    {

      assert(v.size() == nv());
      JointMotion jm = m_constraint * v;
      return m_scaling_factor * jm;
    }

    template<typename S1, int O1>
    SE3ActionReturnType se3Action(const SE3Tpl<S1, O1> & m) const
    {
      return m_scaling_factor * m_constraint.se3Action(m);
    }

    template<typename S1, int O1>
    SE3ActionReturnType se3ActionInverse(const SE3Tpl<S1, O1> & m) const
    {
      return m_scaling_factor * m_constraint.se3ActionInverse(m);
    }

    int nv_impl() const
    {
      return m_constraint.nv();
    }

    struct TransposeConst
    {
      const ScaledJointMotionSubspaceTpl & ref;
      TransposeConst(const ScaledJointMotionSubspaceTpl & ref)
      : ref(ref)
      {
      }

      template<typename Derived>
      // typename ConstraintForceOp<ScaledJointMotionSubspaceTpl, Derived>::ReturnType
      JointForce operator*(const ForceDense<Derived> & f) const
      {
        return ref.m_scaling_factor * (ref.m_constraint.transpose() * f);
      }

      /// [CRBA]  MatrixBase operator* (RefConstraint::Transpose S, ForceSet::Block)
      template<typename Derived>
      typename ConstraintForceSetOp<ScaledJointMotionSubspaceTpl, Derived>::ReturnType
      operator*(const Eigen::MatrixBase<Derived> & F) const
      {
        return ref.m_scaling_factor * (ref.m_constraint.transpose() * F);
      }

    }; // struct TransposeConst

    TransposeConst transpose() const
    {
      return TransposeConst(*this);
    }

    const DenseBase & matrix_impl() const
    {
      S = m_scaling_factor * m_constraint.matrix_impl();
      return S;
    }

    DenseBase & matrix_impl()
    {
      S = m_scaling_factor * m_constraint.matrix_impl();
      return S;
    }

    template<typename MotionDerived>
    typename MotionAlgebraAction<ScaledJointMotionSubspaceTpl, MotionDerived>::ReturnType
    motionAction(const MotionDense<MotionDerived> & m) const
    {
      return m_scaling_factor * m_constraint.motionAction(m);
    }

    inline const Scalar & scaling() const
    {
      return m_scaling_factor;
    }
    inline Scalar & scaling()
    {
      return m_scaling_factor;
    }

    inline const RefJointMotionSubspace & constraint() const
    {
      return m_constraint.derived();
    }
    inline RefJointMotionSubspace & constraint()
    {
      return m_constraint.derived();
    }

    bool isEqual(const ScaledJointMotionSubspaceTpl & other) const
    {
      return m_constraint == other.m_constraint && m_scaling_factor == other.m_scaling_factor;
    }

  protected:
    RefJointMotionSubspace m_constraint;
    Scalar m_scaling_factor;
    mutable DenseBase S;
  }; // struct ScaledJointMotionSubspaceTpl

  template<typename S1, int O1, typename S2, int O2, int MD2>
  struct MultiplicationOp<InertiaTpl<S1, O1>, ScaledJointMotionSubspaceTpl<S2, O2, MD2>>
  {
    typedef InertiaTpl<S1, O1> Inertia;
    typedef ScaledJointMotionSubspaceTpl<S2, O2, MD2> Constraint;
    typedef typename Constraint::Scalar Scalar;

    typedef Eigen::Matrix<S2, 6, Eigen::Dynamic, O2, 6, MD2> ReturnType;
  };

  /* [CRBA] ForceSet operator* (Inertia Y,Constraint S) */
  namespace impl
  {
    template<typename S1, int O1, typename S2, int O2, int MD2>
    struct LhsMultiplicationOp<InertiaTpl<S1, O1>, ScaledJointMotionSubspaceTpl<S2, O2, MD2>>
    {
      typedef InertiaTpl<S1, O1> Inertia;
      typedef ScaledJointMotionSubspaceTpl<S2, O2, MD2> Constraint;
      typedef typename MultiplicationOp<Inertia, Constraint>::ReturnType ReturnType;

      static inline ReturnType run(const Inertia & Y, const Constraint & scaled_constraint)
      {
        return scaled_constraint.scaling() * (Y * scaled_constraint.constraint());
      }
    };
  } // namespace impl

  template<typename M6Like, typename S2, int O2, int MD2>
  struct MultiplicationOp<Eigen::MatrixBase<M6Like>, ScaledJointMotionSubspaceTpl<S2, O2, MD2>>
  {
    typedef ScaledJointMotionSubspaceTpl<S2, O2, MD2> MotionSubspace;
    typedef Eigen::Matrix<S2, 6, Eigen::Dynamic, O2, 6, MD2> ReturnType;
  };

  /* [ABA] operator* (Inertia Y,Constraint S) */
  namespace impl
  {
    template<typename M6Like, typename S2, int O2, int MD2>
    struct LhsMultiplicationOp<Eigen::MatrixBase<M6Like>, ScaledJointMotionSubspaceTpl<S2, O2, MD2>>
    {
      typedef ScaledJointMotionSubspaceTpl<S2, O2, MD2> Constraint;
      typedef
        typename MultiplicationOp<Eigen::MatrixBase<M6Like>, Constraint>::ReturnType ReturnType;

      static inline ReturnType
      run(const Eigen::MatrixBase<M6Like> & Y, const Constraint & scaled_constraint)
      {
        return scaled_constraint.scaling() * (Y.derived() * scaled_constraint.constraint());
      }
    };
  } // namespace impl

  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  struct JointMimicTpl;
  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  struct JointModelMimicTpl;
  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  struct JointDataMimicTpl;

  template<typename _Scalar, int _Options, template<typename S, int O> class JointCollectionTpl>
  struct traits<JointMimicTpl<_Scalar, _Options, JointCollectionTpl>>
  {
    typedef _Scalar Scalar;

    enum
    {
      Options = _Options,
      NQ = Eigen::Dynamic,
      NV = Eigen::Dynamic,
      NJ = Eigen::Dynamic,
      MaxNJ = 6
    };

    typedef JointCollectionTpl<Scalar, Options> JointCollection;
    typedef JointDataMimicTpl<Scalar, Options, JointCollectionTpl> JointDataDerived;
    typedef JointModelMimicTpl<Scalar, Options, JointCollectionTpl> JointModelDerived;
    typedef ScaledJointMotionSubspaceTpl<Scalar, Options, MaxNJ> Constraint_t;
    typedef SE3Tpl<Scalar, Options> Transformation_t;

    typedef MotionTpl<Scalar, Options> Motion_t;
    typedef MotionTpl<Scalar, Options> Bias_t;

    // [ABA]
    typedef Eigen::Matrix<Scalar, 6, Eigen::Dynamic, Options> U_t;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options> D_t;
    typedef Eigen::Matrix<Scalar, 6, Eigen::Dynamic, Options> UD_t;

    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options> ConfigVector_t;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options> TangentVector_t;

    typedef boost::mpl::false_ is_mimicable_t;

    PINOCCHIO_JOINT_DATA_BASE_ACCESSOR_DEFAULT_RETURN_TYPE
  };

  template<typename _Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  struct traits<JointDataMimicTpl<_Scalar, Options, JointCollectionTpl>>
  {
    typedef JointMimicTpl<_Scalar, Options, JointCollectionTpl> JointDerived;
    typedef _Scalar Scalar;
  };

  template<typename _Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  struct traits<JointModelMimicTpl<_Scalar, Options, JointCollectionTpl>>
  {
    typedef JointMimicTpl<_Scalar, Options, JointCollectionTpl> JointDerived;
    typedef _Scalar Scalar;
  };

  template<typename _Scalar, int _Options, template<typename S, int O> class JointCollectionTpl>
  struct JointDataMimicTpl
  : public JointDataBase<JointDataMimicTpl<_Scalar, _Options, JointCollectionTpl>>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef JointDataBase<JointDataMimicTpl> Base;
    typedef JointMimicTpl<_Scalar, _Options, JointCollectionTpl> JointDerived;
    PINOCCHIO_JOINT_DATA_TYPEDEF_TEMPLATE(JointDerived);

    typedef JointDataTpl<_Scalar, _Options, JointCollectionTpl> RefJointData;
    typedef typename RefJointData::JointDataVariant RefJointDataVariant;

    JointDataMimicTpl()
    : m_scaling((Scalar)0)
    , S((Scalar)0)
    {
      joint_q.resize(0, 1);
      joint_q_transformed.resize(0, 1);
      joint_v.resize(0, 1);
      joint_v_transformed.resize(0, 1);
    }

    // JointDataMimicTpl(const JointDataMimicTpl & other)
    // { *this = other; }

    // JointDataMimicTpl(const RefJointDataVariant & jdata,
    //                const Scalar & scaling,
    //                const int & nq,
    //                const int & nv)
    // : m_jdata_ref(jdata)
    // , m_scaling(scaling)
    // , S(m_jdata_ref.S(),scaling)
    // {

    // }

    JointDataMimicTpl(
      const RefJointData & jdata, const Scalar & scaling, const int & nq, const int & nv)
    : m_jdata_ref(checkMimic(jdata.derived()))
    , m_scaling(scaling)
    , S(m_jdata_ref.S(), scaling)
    {
      joint_q.resize(nq, 1);
      joint_q_transformed.resize(nq, 1);
      joint_v.resize(nv, 1);
      joint_v_transformed.resize(nv, 1);
    }

    JointDataMimicTpl & operator=(const JointDataMimicTpl & other)
    {
      m_jdata_ref = other.m_jdata_ref;
      m_scaling = other.m_scaling;
      joint_q = other.joint_q;
      joint_q_transformed = other.joint_q_transformed;
      joint_v = other.joint_v;
      joint_v_transformed = other.joint_v_transformed;
      S = Constraint_t(m_jdata_ref.S(), other.m_scaling);
      return *this;
    }

    using Base::isEqual;
    bool isEqual(const JointDataMimicTpl & other) const
    {
      return Base::isEqual(other) && m_jdata_ref == other.m_jdata_ref
             && m_scaling == other.m_scaling && joint_q == other.joint_q
             && joint_q_transformed == other.joint_q_transformed && joint_v == other.joint_v
             && joint_v_transformed == other.joint_v_transformed;
    }

    static std::string classname()
    {
      return std::string("JointDataMimic");
    }

    std::string shortname() const
    {
      return classname();
    }

    // // Accessors
    ConstraintTypeConstRef S_accessor() const
    {
      return S;
    }
    ConstraintTypeRef S_accessor()
    {
      return S;
    }

    TansformTypeConstRef M_accessor() const
    {
      M_ = m_jdata_ref.M();
      return M_;
    }
    TansformTypeRef M_accessor()
    {
      // assert(false && "Changes to non const ref on mimic joints won't be taken into account. Use
      // const ref");
      M_ = m_jdata_ref.M();
      return M_;
    }

    MotionTypeConstRef v_accessor() const
    {
      v_ = m_jdata_ref.v();
      return v_;
    }
    MotionTypeRef v_accessor()
    {
      // assert(false && "Changes to non const ref on mimic joints won't be taken into account. Use
      // const ref");
      v_ = m_jdata_ref.v();
      return v_;
    }

    BiasTypeConstRef c_accessor() const
    {
      c_ = m_jdata_ref.c();
      return c_;
    }
    BiasTypeRef c_accessor()
    {
      // assert(false && "Changes to non const ref on mimic joints won't be taken into account. Use
      // const ref");
      c_ = m_jdata_ref.c();
      return c_;
    }

    UTypeConstRef U_accessor() const
    {
      U_ = m_jdata_ref.U();
      return U_;
    }
    UTypeRef U_accessor()
    {
      // assert(false && "Changes to non const ref on mimic joints won't be taken into account. Use
      // const ref");
      U_ = m_jdata_ref.U();
      return U_;
    }

    DTypeConstRef Dinv_accessor() const
    {
      Dinv_ = m_jdata_ref.Dinv();
      return Dinv_;
    }
    DTypeRef Dinv_accessor()
    {
      // assert(false && "Changes to non const ref on mimic joints won't be taken into account. Use
      // const ref");
      Dinv_ = m_jdata_ref.Dinv();
      return Dinv_;
    }

    UDTypeConstRef UDinv_accessor() const
    {
      UDinv_ = m_jdata_ref.UDinv();
      return UDinv_;
    }
    UDTypeRef UDinv_accessor()
    {
      // assert(false && "Changes to non const ref on mimic joints won't be taken into account. Use
      // const ref");
      UDinv_ = m_jdata_ref.UDinv();
      return UDinv_;
    }

    DTypeConstRef StU_accessor() const
    {
      StU_ = m_jdata_ref.StU();
      return StU_;
    }
    DTypeRef StU_accessor()
    {
      // assert(false && "Changes to non const ref on mimic joints won't be taken into account. Use
      // const ref");
      StU_ = m_jdata_ref.StU();
      return StU_;
    }

    friend struct JointModelMimicTpl<_Scalar, _Options, JointCollectionTpl>;

    const RefJointData & jdata() const
    {
      return m_jdata_ref;
    }
    RefJointData & jdata()
    {
      return m_jdata_ref;
    }

    const Scalar & scaling() const
    {
      return m_scaling;
    }
    Scalar & scaling()
    {
      return m_scaling;
    }

    ConfigVector_t & joint_q_accessor()
    {
      return joint_q;
    }
    const ConfigVector_t & joint_q_accessor() const
    {
      return joint_q;
    }

    ConfigVector_t & q_transformed()
    {
      return joint_q_transformed;
    }
    const ConfigVector_t & q_transformed() const
    {
      return joint_q_transformed;
    }

    TangentVector_t & joint_v_accessor()
    {
      return joint_v;
    }
    const TangentVector_t & joint_v_accessor() const
    {
      return joint_v;
    }

    TangentVector_t & v_transformed()
    {
      return joint_v_transformed;
    }
    const TangentVector_t & v_transformed() const
    {
      return joint_v_transformed;
    }

  protected:
    RefJointData m_jdata_ref;
    Scalar m_scaling;

    /// \brief original configuration vector
    ConfigVector_t joint_q;
    /// \brief Transformed configuration vector
    ConfigVector_t joint_q_transformed;
    /// \brief original velocity vector
    TangentVector_t joint_v;
    /// \brief Transform velocity vector.
    TangentVector_t joint_v_transformed;

  public:
    // data
    Constraint_t S;

  protected:
    /// \brief Buffer variable for accessors to return references
    mutable Transformation_t M_;
    mutable Motion_t v_;
    mutable Bias_t c_;
    mutable U_t U_;
    mutable D_t Dinv_;
    mutable UD_t UDinv_;
    mutable D_t StU_;

  }; // struct JointDataMimicTpl

  template<
    typename NewScalar,
    typename Scalar,
    int Options,
    template<typename S, int O>
    class JointCollectionTpl>
  struct CastType<NewScalar, JointModelMimicTpl<Scalar, Options, JointCollectionTpl>>
  {
    typedef JointModelMimicTpl<NewScalar, Options, JointCollectionTpl> type;
  };

  template<typename _Scalar, int _Options, template<typename S, int O> class JointCollectionTpl>
  struct JointModelMimicTpl
  : public JointModelBase<JointModelMimicTpl<_Scalar, _Options, JointCollectionTpl>>
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef JointModelBase<JointModelMimicTpl> Base;
    typedef JointMimicTpl<_Scalar, _Options, JointCollectionTpl> JointDerived;
    PINOCCHIO_JOINT_TYPEDEF_TEMPLATE(JointDerived);
    enum
    {
      MaxNJ = traits<JointDerived>::MaxNJ
    };

    typedef JointCollectionTpl<Scalar, Options> JointCollection;
    typedef JointModelTpl<Scalar, Options, JointCollectionTpl> JointModel;
    typedef typename JointModel::JointModelVariant JointModelVariant;

    typedef SE3Tpl<Scalar, Options> SE3;
    typedef MotionTpl<Scalar, Options> Motion;
    typedef InertiaTpl<Scalar, Options> Inertia;

    using Base::id;
    using Base::idx_j;
    using Base::idx_q;
    using Base::idx_v;
    using Base::nj;
    using Base::nq;
    using Base::nv;
    using Base::setIndexes;

    JointModelMimicTpl()
    {
    }

    template<typename JointModel>
    JointModelMimicTpl(
      const JointModelBase<JointModel> & jmodel, const Scalar & scaling, const Scalar & offset)
    : JointModelMimicTpl(jmodel, jmodel, scaling, offset)
    {
    }

    template<typename JointModelMimicking, typename JointModelMimicked>
    JointModelMimicTpl(
      const JointModelBase<JointModelMimicking> & jmodel_mimicking,
      const JointModelBase<JointModelMimicked> & jmodel_mimicked,
      const Scalar & scaling,
      const Scalar & offset)
    : m_jmodel_ref(checkMimic((JointModel)jmodel_mimicking.derived()))
    , m_scaling(scaling)
    , m_offset(offset)
    {
      assert(jmodel_mimicking.nq() == jmodel_mimicked.nq());
      assert(jmodel_mimicking.nv() == jmodel_mimicked.nv());
      assert(jmodel_mimicking.nj() == jmodel_mimicked.nj());

      setMimicIndexes(
        jmodel_mimicked.id(), jmodel_mimicked.idx_q(), jmodel_mimicked.idx_v(),
        jmodel_mimicked.idx_j());
    }

    Base & base()
    {
      return *static_cast<Base *>(this);
    }
    const Base & base() const
    {
      return *static_cast<const Base *>(this);
    }

    inline int nq_impl() const
    {
      return 0;
    }
    inline int nv_impl() const
    {
      return 0;
    }
    inline int nj_impl() const
    {
      return m_jmodel_ref.nj();
    }

    void setIndexes_impl(JointIndex id, int /*q*/, int /*v*/, int j)
    {
      Base::i_id = id;
      // When setting the indexes q and v should remain on the mimicked joint
      // Base::i_q = q;
      // Base::i_v = v;
      Base::i_j = j;
    }

    // Specific way for mimic joints to set the mimicked q,v indexes.
    // Used for manipulating tree (e.g. appendModel)
    void setMimicIndexes(JointIndex id, int q, int v, int j)
    {
      // Set idx_q, idx_v to zero so that only sub segment of q,v can be passed to ref joint
      m_jmodel_ref.setIndexes(id, 0, 0, j);
      // idx_q, idx_v kept separately
      Base::i_q = q;
      Base::i_v = v;
    }

    JointDataDerived createData() const
    {

      return JointDataDerived(
        m_jmodel_ref.createData(), scaling(), m_jmodel_ref.nq(), m_jmodel_ref.nv());
    }

    const std::vector<bool> hasConfigurationLimit() const
    {
      return m_jmodel_ref.hasConfigurationLimit();
    }

    const std::vector<bool> hasConfigurationLimitInTangent() const
    {
      return m_jmodel_ref.hasConfigurationLimitInTangent();
    }

    template<typename ConfigVector>
    EIGEN_DONT_INLINE void
    calc(JointDataDerived & jdata, const typename Eigen::MatrixBase<ConfigVector> & qs) const
    {
      jdata.joint_q = qs.segment(idx_q(), m_jmodel_ref.nq());
      configVectorAffineTransform(
        m_jmodel_ref, jdata.joint_q, m_scaling, m_offset, jdata.joint_q_transformed);
      m_jmodel_ref.calc(jdata.m_jdata_ref, jdata.joint_q_transformed);
    }

    template<typename ConfigVector, typename TangentVector>
    EIGEN_DONT_INLINE void calc(
      JointDataDerived & jdata,
      const typename Eigen::MatrixBase<ConfigVector> & qs,
      const typename Eigen::MatrixBase<TangentVector> & vs) const
    {
      jdata.joint_q = qs.segment(idx_q(), m_jmodel_ref.nq());
      jdata.joint_v = vs.segment(idx_v(), m_jmodel_ref.nv());
      configVectorAffineTransform(
        m_jmodel_ref, jdata.joint_q, m_scaling, m_offset, jdata.joint_q_transformed);
      jdata.joint_v_transformed = m_scaling * jdata.joint_v;
      m_jmodel_ref.calc(jdata.m_jdata_ref, jdata.joint_q_transformed, jdata.joint_v_transformed);
    }

    template<typename VectorLike, typename Matrix6Like>
    void calc_aba(
      JointDataDerived & data,
      const Eigen::MatrixBase<VectorLike> & armature,
      const Eigen::MatrixBase<Matrix6Like> & I,
      const bool update_I) const
    {
      // TODO: fixme
      assert(
        false
        && "Joint Mimic is not supported for aba yet. Remove it from your model if you want to use "
           "this function");
      m_jmodel_ref.calc_aba(
        data.m_jdata_ref, armature, PINOCCHIO_EIGEN_CONST_CAST(Matrix6Like, I), update_I);
    }

    static std::string classname()
    {
      return std::string("JointModelMimic");
    }

    std::string shortname() const
    {
      return classname();
    }

    /// \returns An expression of *this with the Scalar type casted to NewScalar.
    template<typename NewScalar>
    typename CastType<NewScalar, JointModelMimicTpl>::type cast() const
    {
      typedef typename CastType<NewScalar, JointModelMimicTpl>::type ReturnType;

      ReturnType res(
        m_jmodel_ref.template cast<NewScalar>(), ScalarCast<NewScalar, Scalar>::cast(m_scaling),
        ScalarCast<NewScalar, Scalar>::cast(m_offset));
      res.setIndexes(id(), idx_q(), idx_v(), idx_j());
      return res;
    }

    const JointModel & jmodel() const
    {
      return m_jmodel_ref;
    }
    JointModel & jmodel()
    {
      return m_jmodel_ref;
    }

    const Scalar & scaling() const
    {
      return m_scaling;
    }
    Scalar & scaling()
    {
      return m_scaling;
    }

    const Scalar & offset() const
    {
      return m_offset;
    }
    Scalar & offset()
    {
      return m_offset;
    }

  protected:
    // data
    JointModel m_jmodel_ref;
    Scalar m_scaling, m_offset;

  public:
    /* Acces to dedicated segment in robot config space.  */
    // Const access
    template<typename D>
    typename SizeDepType<NQ>::template SegmentReturn<D>::ConstType
    jointConfigFromDofSelector_impl(const Eigen::MatrixBase<D> & a) const
    {
      return SizeDepType<NQ>::segment(a.derived(), idx_q(), m_jmodel_ref.nq());
    }

    // Non-const access
    template<typename D>
    typename SizeDepType<NQ>::template SegmentReturn<D>::Type
    jointConfigFromDofSelector_impl(Eigen::MatrixBase<D> & a) const
    {
      return SizeDepType<NQ>::segment(a.derived(), idx_q(), m_jmodel_ref.nq());
    }

    /* Acces to dedicated columns in a ForceSet or MotionSet matrix.*/
    // Const access
    template<typename D>
    typename SizeDepType<NV>::template ColsReturn<D>::ConstType
    jointVelCols_impl(const Eigen::MatrixBase<D> & A) const
    {
      return SizeDepType<NV>::middleCols(A.derived(), idx_v(), m_jmodel_ref.nv());
    }

    // Non-const access
    template<typename D>
    typename SizeDepType<NV>::template ColsReturn<D>::Type
    jointVelCols_impl(Eigen::MatrixBase<D> & A) const
    {
      return SizeDepType<NV>::middleCols(A.derived(), idx_v(), m_jmodel_ref.nv());
    }

    /* Acces to dedicated rows in a matrix.*/
    // Const access
    template<typename D>
    typename SizeDepType<NV>::template RowsReturn<D>::ConstType
    joinVelRows_impl(const Eigen::MatrixBase<D> & A) const
    {
      return SizeDepType<NV>::middleRows(A.derived(), idx_v(), m_jmodel_ref.nv());
    }

    // Non-const access
    template<typename D>
    typename SizeDepType<NV>::template RowsReturn<D>::Type
    jointVelRows_impl(Eigen::MatrixBase<D> & A) const
    {
      return SizeDepType<NV>::middleRows(A.derived(), idx_v(), m_jmodel_ref.nv());
    }

    // /// \brief Returns a block of dimension nv()xnv() located at position idx_v(),idx_v() in the
    // matrix Mat
    // // Const access
    template<typename D>
    typename SizeDepType<NV>::template BlockReturn<D>::ConstType
    jointVelBlock_impl(const Eigen::MatrixBase<D> & Mat) const
    {
      return SizeDepType<NV>::block(
        Mat.derived(), idx_v(), idx_v(), m_jmodel_ref.nv(), m_jmodel_ref.nv());
    }

    // Non-const access
    template<typename D>
    typename SizeDepType<NV>::template BlockReturn<D>::Type
    jointVelBlock_impl(Eigen::MatrixBase<D> & Mat) const
    {
      return SizeDepType<NV>::block(
        Mat.derived(), idx_v(), idx_v(), m_jmodel_ref.nv(), m_jmodel_ref.nv());
    }

  }; // struct JointModelMimicTpl

} // namespace pinocchio

#include <boost/type_traits.hpp>

namespace boost
{
  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  struct has_nothrow_constructor<
    ::pinocchio::JointModelMimicTpl<Scalar, Options, JointCollectionTpl>>
  : public integral_constant<bool, true>
  {
  };

  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  struct has_nothrow_copy<::pinocchio::JointModelMimicTpl<Scalar, Options, JointCollectionTpl>>
  : public integral_constant<bool, true>
  {
  };

  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  struct has_nothrow_constructor<
    ::pinocchio::JointDataMimicTpl<Scalar, Options, JointCollectionTpl>>
  : public integral_constant<bool, true>
  {
  };

  template<typename Scalar, int Options, template<typename S, int O> class JointCollectionTpl>
  struct has_nothrow_copy<::pinocchio::JointDataMimicTpl<Scalar, Options, JointCollectionTpl>>
  : public integral_constant<bool, true>
  {
  };
} // namespace boost

#endif // ifndef __pinocchio_multibody_joint_mimic_hpp__
