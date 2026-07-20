#include <atomic>
#include <vector>
#include <utility>
#include <cstring>

#include "d3d11_device.h"
#include "d3d11_context_imm.h"
#include "d3d11_context_def.h"
#include "d3d11_cuda.h"

#include "../util/log/log.h"

#include <type_traits>

namespace dxvk {
  
  // H4A TEST — local timestamp helper, matching dxvk_queue.cpp's convention.
  // TEMPORARY — revert once H4a is confirmed or fully killed.
  static int64_t smpDiagSmvNowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  
  template<typename ContextType>
  D3D11DeviceContextExt<ContextType>::D3D11DeviceContextExt(
          ContextType*          pContext)
  : m_ctx(pContext) {
    
  }

  template <typename ContextType>
  void STDMETHODCALLTYPE D3D11DeviceContextExt<ContextType>::SetMultiviewModeNV(
      uint32_t NumViews, BOOL IndependentViewportMask) {
    Logger::warn(str::format("[SMP-DIAG-SMV] t=", smpDiagSmvNowMs(), " ENTER (before lock) this=", (void*)this));
    Logger::warn(str::format("[SMP-DIAG-CALLER] retaddr=", __builtin_return_address(0)));
    D3D10DeviceLock lock = m_ctx->LockContext();
    Logger::warn(str::format("[SMP-DIAG-SMV] t=", smpDiagSmvNowMs(), " LOCK ACQUIRED this=", (void*)this));

    // ~44% of iRacing's calls are redundant no-op re-sets (recon, handoff
    // section 3): dedupe under the context lock, before the CS stream.
    // Reads the one true copy on D3D11CommonContext, not a field of
    // its own - the Ext class already holds m_ctx, so it can read
    // through that pointer instead of keeping a second copy that could
    // drift out of step with it.
    if (NumViews == m_ctx->GetNvMultiviewNumViews() &&
        bool(IndependentViewportMask) ==
            m_ctx->GetNvMultiviewIndependentMask()) {
      Logger::warn(str::format("[SMP-DIAG-SMV] t=", smpDiagSmvNowMs(), " EARLY-EXIT (redundant no-op) this=", (void*)this));
      return;
    }

    m_ctx->SetNvMultiviewToggleState(NumViews, bool(IndependentViewportMask));

    // Budgeted evidence: a handful of transitions per session, not 24k.
    static std::atomic<int32_t> s_logBudget = {8};

    if (s_logBudget.fetch_sub(1, std::memory_order_relaxed) > 0) {
      Logger::info(str::format("SetMultiviewModeNV: numViews=", NumViews,
                               " independentMask=", IndependentViewportMask,
                               " (forwarded to CS)"));
    }

    m_ctx->EmitCs(
        [cNumViews = NumViews,
         cIndependentMask = bool(IndependentViewportMask)](DxvkContext *ctx) {
          ctx->setNvMultiviewState(cNumViews, cIndependentMask);
        });

    // H3 TEST 2 (THREAD_B_H3_PLAN.md) — forced-sync probe. Only compiles/runs
    // for the immediate context, since SynchronizeCsThread doesn't exist on
    // the deferred context. Closes the two-whiteboard window immediately
    // after every genuine (non-redundant) count change, before anything
    // else gets queued against the new value. TEMPORARY — revert after
    // reading the result, per BUILD_TEST_REVERT_WORKFLOW.md.
    if constexpr (std::is_same_v<ContextType, D3D11ImmediateContext>) {
      Logger::warn(str::format("[SMP-DIAG-H3SYNC] ENTER this=", (void*)this));
      m_ctx->SynchronizeCsThread(DxvkCsThread::SynchronizeAll);
      Logger::warn(str::format("[SMP-DIAG-H3SYNC] EXIT this=", (void*)this));
    }

    Logger::warn(str::format("[SMP-DIAG-SMV] t=", smpDiagSmvNowMs(), " EXIT (normal) this=", (void*)this));
  }

  template<typename ContextType>
  ULONG STDMETHODCALLTYPE D3D11DeviceContextExt<ContextType>::AddRef() {
    return m_ctx->AddRef();
  }
  
  
  template<typename ContextType>
  ULONG STDMETHODCALLTYPE D3D11DeviceContextExt<ContextType>::Release() {
    return m_ctx->Release();
  }
  
  
  template<typename ContextType>
  HRESULT STDMETHODCALLTYPE D3D11DeviceContextExt<ContextType>::QueryInterface(
          REFIID                  riid,
          void**                  ppvObject) {
    return m_ctx->QueryInterface(riid, ppvObject);
  }
  
  
  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11DeviceContextExt<ContextType>::MultiDrawIndirect(
          UINT                    DrawCount,
          ID3D11Buffer*           pBufferForArgs,
          UINT                    ByteOffsetForArgs,
          UINT                    ByteStrideForArgs) {
    D3D10DeviceLock lock = m_ctx->LockContext();
    m_ctx->SetDrawBuffers(pBufferForArgs, nullptr);
    
    if (unlikely(m_ctx->HasDirtyGraphicsBindings()))
      m_ctx->ApplyDirtyGraphicsBindings();

    m_ctx->EmitCs([
      cCount  = DrawCount,
      cOffset = ByteOffsetForArgs,
      cStride = ByteStrideForArgs
    ] (DxvkContext* ctx) {
      ctx->drawIndirect(cOffset, cCount, cStride, false);
    });
  }
  
  
  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11DeviceContextExt<ContextType>::MultiDrawIndexedIndirect(
          UINT                    DrawCount,
          ID3D11Buffer*           pBufferForArgs,
          UINT                    ByteOffsetForArgs,
          UINT                    ByteStrideForArgs) {
    D3D10DeviceLock lock = m_ctx->LockContext();
    m_ctx->SetDrawBuffers(pBufferForArgs, nullptr);

    if (unlikely(m_ctx->HasDirtyGraphicsBindings()))
      m_ctx->ApplyDirtyGraphicsBindings();

    m_ctx->EmitCs([
      cCount  = DrawCount,
      cOffset = ByteOffsetForArgs,
      cStride = ByteStrideForArgs
    ] (DxvkContext* ctx) {
      ctx->drawIndexedIndirect(cOffset, cCount, cStride, false);
    });
  }
  
  
  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11DeviceContextExt<ContextType>::MultiDrawIndirectCount(
          UINT                    MaxDrawCount,
          ID3D11Buffer*           pBufferForCount,
          UINT                    ByteOffsetForCount,
          ID3D11Buffer*           pBufferForArgs,
          UINT                    ByteOffsetForArgs,
          UINT                    ByteStrideForArgs) {
    D3D10DeviceLock lock = m_ctx->LockContext();
    m_ctx->SetDrawBuffers(pBufferForArgs, pBufferForCount);

    if (unlikely(m_ctx->HasDirtyGraphicsBindings()))
      m_ctx->ApplyDirtyGraphicsBindings();

    m_ctx->EmitCs([
      cMaxCount  = MaxDrawCount,
      cArgOffset = ByteOffsetForArgs,
      cCntOffset = ByteOffsetForCount,
      cStride    = ByteStrideForArgs
    ] (DxvkContext* ctx) {
      ctx->drawIndirectCount(cArgOffset, cCntOffset, cMaxCount, cStride);
    });
  }
  
  
  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11DeviceContextExt<ContextType>::MultiDrawIndexedIndirectCount(
          UINT                    MaxDrawCount,
          ID3D11Buffer*           pBufferForCount,
          UINT                    ByteOffsetForCount,
          ID3D11Buffer*           pBufferForArgs,
          UINT                    ByteOffsetForArgs,
          UINT                    ByteStrideForArgs) {
    D3D10DeviceLock lock = m_ctx->LockContext();
    m_ctx->SetDrawBuffers(pBufferForArgs, pBufferForCount);

    if (unlikely(m_ctx->HasDirtyGraphicsBindings()))
      m_ctx->ApplyDirtyGraphicsBindings();

    m_ctx->EmitCs([
      cMaxCount  = MaxDrawCount,
      cArgOffset = ByteOffsetForArgs,
      cCntOffset = ByteOffsetForCount,
      cStride    = ByteStrideForArgs
    ] (DxvkContext* ctx) {
      ctx->drawIndexedIndirectCount(cArgOffset, cCntOffset, cMaxCount, cStride);
    });
  }
  
  
  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11DeviceContextExt<ContextType>::SetDepthBoundsTest(
          BOOL                    Enable,
          FLOAT                   MinDepthBounds,
          FLOAT                   MaxDepthBounds) {
    D3D10DeviceLock lock = m_ctx->LockContext();

    DxvkDepthBounds db = { };
    db.minDepthBounds = Enable ? MinDepthBounds : 0.0f;
    db.maxDepthBounds = Enable ? MaxDepthBounds : 1.0f;
    
    m_ctx->EmitCs([cDepthBounds = db] (DxvkContext* ctx) {
      ctx->setDepthBounds(cDepthBounds);
    });
  }
  
  
  template<typename ContextType>
  void STDMETHODCALLTYPE D3D11DeviceContextExt<ContextType>::SetBarrierControl(
          UINT                    ControlFlags) {
    D3D10DeviceLock lock = m_ctx->LockContext();
    D3D11Device* parent = static_cast<D3D11Device*>(m_ctx->GetParentInterface());
    DxvkBarrierControlFlags flags = parent->GetOptionsBarrierControlFlags();

    if (ControlFlags & D3D11_VK_BARRIER_CONTROL_IGNORE_WRITE_AFTER_WRITE) {
      flags.set(DxvkBarrierControl::ComputeAllowReadWriteOverlap,
                DxvkBarrierControl::GraphicsAllowReadWriteOverlap);
    }

    m_ctx->EmitCs([cFlags = flags] (DxvkContext* ctx) {
      ctx->setBarrierControl(cFlags);
    });
  }


  template<typename ContextType>
  bool STDMETHODCALLTYPE D3D11DeviceContextExt<ContextType>::LaunchCubinShaderNVX(IUnknown* hShader, uint32_t GridX, uint32_t GridY, uint32_t GridZ,
      const void* pParams, uint32_t ParamSize, void* const* pReadResources, uint32_t NumReadResources, void* const* pWriteResources, uint32_t NumWriteResources) {
    D3D10DeviceLock lock = m_ctx->LockContext();

    CubinShaderWrapper* cubinShader = static_cast<CubinShaderWrapper*>(hShader);
    CubinShaderLaunchInfo launchInfo;

    const uint32_t maxResources = NumReadResources + NumWriteResources;
    launchInfo.buffers.reserve(maxResources);
    launchInfo.images.reserve(maxResources);

    for (uint32_t i = 0; i < NumReadResources; i++)
      launchInfo.insertResource(static_cast<ID3D11Resource*>(pReadResources[i]), DxvkAccess::Read);

    for (uint32_t i = 0; i < NumWriteResources; i++)
      launchInfo.insertResource(static_cast<ID3D11Resource*>(pWriteResources[i]), DxvkAccess::Write);

    launchInfo.paramSize = ParamSize;
    launchInfo.params.resize(launchInfo.paramSize);
    std::memcpy(launchInfo.params.data(), pParams, ParamSize);

    launchInfo.cuLaunchConfig[0] = reinterpret_cast<void*>(0x01); // CU_LAUNCH_PARAM_BUFFER_POINTER
    launchInfo.cuLaunchConfig[1] = launchInfo.params.data();
    launchInfo.cuLaunchConfig[2] = reinterpret_cast<void*>(0x02); // CU_LAUNCH_PARAM_BUFFER_SIZE
    launchInfo.cuLaunchConfig[3] = &launchInfo.paramSize; // yes, this actually requires a pointer to a size_t containing the parameter size
    launchInfo.cuLaunchConfig[4] = reinterpret_cast<void*>(0x00); // CU_LAUNCH_PARAM_END

    launchInfo.nvxLaunchInfo.function       = cubinShader->cuFunction();
    launchInfo.nvxLaunchInfo.gridDimX       = GridX;
    launchInfo.nvxLaunchInfo.gridDimY       = GridY;
    launchInfo.nvxLaunchInfo.gridDimZ       = GridZ;
    launchInfo.nvxLaunchInfo.blockDimX      = cubinShader->blockDim().width;
    launchInfo.nvxLaunchInfo.blockDimY      = cubinShader->blockDim().height;
    launchInfo.nvxLaunchInfo.blockDimZ      = cubinShader->blockDim().depth;
    launchInfo.nvxLaunchInfo.sharedMemBytes = 0;
    launchInfo.nvxLaunchInfo.paramCount     = 0;
    launchInfo.nvxLaunchInfo.pParams        = nullptr;
    launchInfo.nvxLaunchInfo.extraCount     = 1;
    launchInfo.nvxLaunchInfo.pExtras        = launchInfo.cuLaunchConfig.data();

    launchInfo.shader = cubinShader;

    /* Need to capture by value in case this gets called from a deferred context */
    m_ctx->AddCost(GpuCostEstimate::Dispatch);

    m_ctx->EmitCs([cLaunchInfo = std::move(launchInfo)] (DxvkContext* ctx) {
      ctx->launchCuKernelNVX(cLaunchInfo.nvxLaunchInfo, cLaunchInfo.buffers, cLaunchInfo.images);
    });

    // Track resource usage as necessary
    for (uint32_t i = 0; i < NumReadResources; i++)
      m_ctx->TrackResourceSequenceNumber(static_cast<ID3D11Resource*>(pReadResources[i]));

    for (uint32_t i = 0; i < NumWriteResources; i++)
      m_ctx->TrackResourceSequenceNumber(static_cast<ID3D11Resource*>(pWriteResources[i]));

    return true;
  }


  template class D3D11DeviceContextExt<D3D11DeferredContext>;
  template class D3D11DeviceContextExt<D3D11ImmediateContext>;

}
